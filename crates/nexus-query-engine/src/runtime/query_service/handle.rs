#[cfg(feature = "cugraph")]
use super::super::context::{ReservedResidentGraphHit, warm_graph_reservation_error};
use super::{
    super::{
        context::{QueryResourceAnchor, QueryResourceEpoch, QueryServiceResourceEpoch},
        memory_domain::{create_grant_memory_domain, create_grant_reservation_arena},
    },
    *,
};

#[derive(Debug)]
pub(super) struct QueryHandleLifetime {
    pub(super) service: Arc<QueryServiceInner>,
    pub(super) tenant_id: TenantId,
    pub(super) admission_request_id: AdmissionRequestId,
    observation: RuntimeObservationHandle,
    observation_owner: Option<RuntimeObservationOwner>,
    observation_activity: Mutex<Option<QueryActivityLease>>,
    service_epoch_user: Mutex<Option<QueryServiceEpochUser>>,
    device_ordinal: usize,
    release_requested: AtomicBool,
    released: AtomicBool,
    query_epoch: QueryResourceEpoch,
    root_context: Mutex<Option<ExecutionContext>>,
    memory_grant: Mutex<Option<MemoryGrant>>,
    device_slot: Mutex<Option<DeviceActiveQuerySlot>>,
    #[cfg(feature = "cugraph")]
    warm_graph_reservation: Mutex<Option<WarmGraphReservation>>,
    #[cfg(feature = "cugraph")]
    warm_graph_priced: bool,
}

impl QueryHandleLifetime {
    pub(super) fn new(
        service: Arc<QueryServiceInner>,
        tenant_id: TenantId,
        admission_request_id: AdmissionRequestId,
        observation: RuntimeObservationHandle,
        observation_owner: Option<RuntimeObservationOwner>,
        observation_activity: Option<QueryActivityLease>,
        memory_grant: MemoryGrant,
        device_slot: Option<DeviceActiveQuerySlot>,
        #[cfg(feature = "cugraph")] warm_graph_reservation: Option<WarmGraphReservation>,
    ) -> Arc<Self> {
        let device_ordinal =
            device_slot.as_ref().map(DeviceActiveQuerySlot::device_ordinal).unwrap_or(service.config.device_ordinal);
        #[cfg(feature = "cugraph")]
        let warm_graph_priced = warm_graph_reservation.is_some();
        let service_epoch_user = service.service_epoch.retain_existing_user();
        Arc::new_cyclic(|weak: &Weak<QueryHandleLifetime>| {
            let weak_lifetime = weak.clone();
            let on_query_idle = Arc::new(move || {
                if let Some(lifetime) = weak_lifetime.upgrade() {
                    lifetime.release_if_ready();
                }
            });
            Self {
                service,
                tenant_id,
                admission_request_id,
                observation,
                observation_owner,
                observation_activity: Mutex::new(observation_activity),
                service_epoch_user: Mutex::new(Some(service_epoch_user)),
                device_ordinal,
                release_requested: AtomicBool::new(false),
                released: AtomicBool::new(false),
                query_epoch: QueryResourceEpoch::new(Some(on_query_idle)),
                root_context: Mutex::new(None),
                memory_grant: Mutex::new(Some(memory_grant)),
                device_slot: Mutex::new(device_slot),
                #[cfg(feature = "cugraph")]
                warm_graph_reservation: Mutex::new(warm_graph_reservation),
                #[cfg(feature = "cugraph")]
                warm_graph_priced,
            }
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

    fn acquire_resource_anchor(self: &Arc<Self>, operation: &'static str) -> Result<Arc<QueryResourceAnchor>> {
        if self.release_requested.load(Ordering::Acquire) || self.released.load(Ordering::Acquire) {
            return Err(self.released_error());
        }
        let owner: Arc<dyn Send + Sync> = self.clone();
        self.query_epoch.acquire(owner, operation).map_err(|_error| self.released_error())
    }

    fn request_release(&self) {
        self.release_requested.store(true, Ordering::Release);
        self.query_epoch.invalidate();
        self.release_if_ready();
    }

    fn release_if_ready(&self) {
        if !self.release_requested.load(Ordering::Acquire) || !self.query_epoch.is_idle() {
            return;
        }
        if self.released.swap(true, Ordering::AcqRel) {
            return;
        }
        let root_context = take_owned_for_cleanup(&self.root_context, "root context");
        let memory_grant = take_owned_for_cleanup(&self.memory_grant, "memory grant");
        let device_slot = take_owned_for_cleanup(&self.device_slot, "device slot");
        let observation_activity = take_owned_for_cleanup(&self.observation_activity, "observation activity");
        let service_epoch_user = take_owned_for_cleanup(&self.service_epoch_user, "service epoch user");
        #[cfg(feature = "cugraph")]
        let warm_graph_reservation = take_owned_for_cleanup(&self.warm_graph_reservation, "warm graph reservation");
        let release = QueryHandleRelease {
            service: Arc::clone(&self.service),
            tenant_id: self.tenant_id.clone(),
            admission_request_id: self.admission_request_id,
            observation: self.observation.clone(),
            observation_owner: self.observation_owner.clone(),
            observation_activity,
            service_epoch_user,
            device_ordinal: self.device_ordinal,
            root_context,
            memory_grant,
            device_slot,
            #[cfg(feature = "cugraph")]
            warm_graph_reservation,
        };
        if let Some(owner) = &release.observation_owner {
            owner.begin_query_cleanup();
        }
        let pool = self
            .service
            .resources
            .as_ref()
            .and_then(|resources| resources.gpu_execution_pool_for_device(self.device_ordinal).ok());
        match pool {
            Some(pool) => {
                let observation = self.observation.clone();
                pool.spawn_observed_with_completion(
                    GpuExecutionJobKind::Cleanup,
                    observation,
                    async move { release.finish() },
                    || {},
                );
            },
            None => {
                release.finish();
            },
        }
    }

    fn released_error(&self) -> QueryError {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "nexus_query_engine.query_service.context",
            "query service handle was released before context acquisition",
        )
        .with_fact("tenant_id", self.tenant_id.as_str())
        .with_fact("admission_request_id", self.admission_request_id)
        .with_fact("reason", "query_handle_released")
    }
}

fn take_owned_for_cleanup<T>(owned: &Mutex<Option<T>>, resource: &'static str) -> Option<T> {
    match owned.lock() {
        Ok(mut owned) => owned.take(),
        Err(error) => {
            eprintln!("[nexus-query-engine] warning: recovering poisoned query service {resource}");
            error.into_inner().take()
        },
    }
}

#[cfg(feature = "cugraph")]
fn require_warm_graph_reservation<T>(reservation: Option<T>, warm_graph_priced: bool) -> Result<Option<T>> {
    if reservation.is_none() && warm_graph_priced {
        return Err(warm_graph_reservation_error("stream_and_pin_reservation_already_consumed"));
    }
    Ok(reservation)
}

struct QueryHandleRelease {
    service: Arc<QueryServiceInner>,
    tenant_id: TenantId,
    admission_request_id: AdmissionRequestId,
    observation: RuntimeObservationHandle,
    observation_owner: Option<RuntimeObservationOwner>,
    observation_activity: Option<QueryActivityLease>,
    service_epoch_user: Option<QueryServiceEpochUser>,
    device_ordinal: usize,
    root_context: Option<ExecutionContext>,
    memory_grant: Option<MemoryGrant>,
    device_slot: Option<DeviceActiveQuerySlot>,
    #[cfg(feature = "cugraph")]
    warm_graph_reservation: Option<WarmGraphReservation>,
}

impl QueryHandleRelease {
    fn finish(mut self) {
        // Remove the adaptive-growth clone before dropping our grant, while
        // retaining the active slot until every GPU-owned resource is gone.
        // The final MemoryGrant drop invokes observers that reacquire this
        // mutex, so it must never happen while the mutex is held.
        match self.service.admission.lock() {
            Ok(mut state) => {
                state.active_adaptive_grants.remove(&self.admission_request_id);
                state.cleanup_in_progress = state.cleanup_in_progress.saturating_add(1);
            },
            Err(_error) => {
                eprintln!("[nexus-query-engine] warning: failed to remove poisoned query service adaptive grant state");
            },
        }
        if let Some(grant) = self.memory_grant.as_ref() {
            let live_reserved_bytes = grant.live_reserved_bytes();
            if live_reserved_bytes != 0
                && let Err(error) = grant.try_shrink_to(live_reserved_bytes)
            {
                eprintln!(
                    "[nexus-query-engine] warning: failed to shrink query service memory grant on release: {error}"
                );
            }
        }
        let root_context = self.root_context.take();
        if root_context.as_ref().is_some_and(ExecutionContext::has_root_stream_lease) {
            let root_context = root_context.expect("root context with a stream lease is present");
            let installed = root_context.install_root_stream_cleanup_completion(move || {
                self.complete_after_stream_cleanup();
            });
            debug_assert!(installed, "root stream cleanup completion is installed exactly once");
            drop(root_context);
            return;
        }
        drop(root_context);
        self.complete_after_stream_cleanup();
    }

    fn complete_after_stream_cleanup(mut self) {
        #[cfg(feature = "cugraph")]
        drop(self.warm_graph_reservation.take());
        drop(self.memory_grant.take());
        drop(self.device_slot.take());
        // Publish query quiescence only after this handle no longer retains
        // the service epoch. `close_and_wait_for_quiescence` uses the active
        // query count as its completion fence, so waking it earlier could let
        // a replacement race the final governor-claim release.
        drop(self.service_epoch_user.take());

        let queue_depth = match self.service.admission.lock() {
            Ok(mut state) => {
                state.active_queries = state.active_queries.saturating_sub(1);
                state.cleanup_in_progress = state.cleanup_in_progress.saturating_sub(1);
                if state.queued.is_empty() {
                    QueryService { inner: Arc::clone(&self.service) }.grow_active_adaptive_grants_locked(&mut state);
                }
                state.queued.len()
            },
            Err(_error) => {
                eprintln!("[nexus-query-engine] warning: failed to release poisoned query service admission state");
                0
            },
        };
        notify_shared_device_admission_available(self.device_ordinal);
        self.service.notify_admission_available();
        record_admission_lifecycle(
            &self.observation,
            self.admission_request_id,
            &self.tenant_id,
            AdmissionLifecycleState::Released,
            queue_depth,
            Some(self.device_ordinal),
            None,
        );
        drop(self.observation_activity.take());
        if let Some(owner) = self.observation_owner.take() {
            let _ = owner.finish_query_cleanup(RuntimeTerminalOutcome::Dropped);
        }
    }
}

/// RAII admission handle for one tenant query.
#[derive(Debug)]
pub struct QueryHandle {
    pub(super) service: Arc<QueryServiceInner>,
    pub(super) tenant_id: TenantId,
    pub(super) admission_request_id: AdmissionRequestId,
    pub(super) admission_sequence: u64,
    pub(super) admission_wait: Duration,
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

    /// Lakehouse cache-fill resource policy assigned to this admitted query.
    pub fn lakehouse_cache_fill_config(&self) -> LakehouseCacheFillConfig {
        self.service.config.lakehouse_cache_fill.clone()
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

    /// Acquires a tenant-bound execution context without blocking the caller's async runtime.
    pub async fn acquire_context_async(self: &Arc<Self>) -> Result<TenantQueryContext> {
        self.acquire_context_with_cancellation_async(ExecutionCancellation::new()).await
    }

    /// Acquires a tenant-bound execution context on this query's device executor.
    pub async fn acquire_context_with_cancellation_async(
        self: &Arc<Self>,
        cancellation: ExecutionCancellation,
    ) -> Result<TenantQueryContext> {
        let pool = self.gpu_execution_pool()?;
        let observation = self.lifetime.observation.clone();
        let handle = Arc::clone(self);
        pool.run_observed_scoped(GpuExecutionJobKind::ContextSetup, observation, move |stream_scope| async move {
            handle.acquire_context_with_cancellation_and_stream_scope(cancellation, stream_scope)
        })
        .await?
    }

    /// Dedicated executor for synchronous GPU work on this query's selected device.
    pub fn gpu_execution_pool(&self) -> Result<GpuExecutionPool> {
        let operation = "nexus_query_engine.query_service.gpu_execution_pool";
        let query_anchor = self.lifetime.acquire_resource_anchor(operation)?;
        let Some(resources) = &self.service.resources else {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                operation,
                "query service was constructed without runtime resources",
            ));
        };
        let pool = resources.gpu_execution_pool_for_device(self.device_ordinal)?;
        let resource_epoch =
            QueryServiceResourceEpoch::acquire(&self.service.service_epoch, Some(&query_anchor), operation)?;
        Ok(pool.with_resource_epoch(resource_epoch))
    }

    /// Acquires a tenant-bound execution context that observes caller-supplied
    /// cooperative cancellation.
    pub fn acquire_context_with_cancellation(&self, cancellation: ExecutionCancellation) -> Result<TenantQueryContext> {
        self.acquire_context_with_cancellation_and_stream_scope(
            cancellation,
            crate::observability::RuntimeStreamLeaseScope::default(),
        )
    }

    fn acquire_context_with_cancellation_and_stream_scope(
        &self,
        cancellation: ExecutionCancellation,
        stream_scope: crate::observability::RuntimeStreamLeaseScope,
    ) -> Result<TenantQueryContext> {
        let operation = "nexus_query_engine.query_service.context";
        let query_anchor = self.lifetime.acquire_resource_anchor(operation)?;
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
        if let Some(template) = root_context.as_ref().cloned() {
            // Concurrent fragments under one admitted query must not share the
            // root stream: cuDF rejects a second live query-domain budget token
            // on the same stream. Bind each secondary acquire to a dedicated
            // service stream while preserving the shared grant/domain.
            drop(root_context);
            let stream_acquire_deadline =
                Instant::now().checked_add(self.service.config.admission_wait_timeout).unwrap_or_else(Instant::now);
            let stream_lease = device.stream_pool().acquire_observed_with_cancellation_deadline(
                &cancellation,
                stream_acquire_deadline,
                "nexus_query_engine.query_service.context.stream_acquire",
                self.lifetime.observation.clone(),
                stream_scope,
            )?;
            validate_service_stream_pool_device(device.stream_pool(), stream_lease.device())?;
            let context =
                template.with_owned_stream_lease(stream_lease)?.into_context().with_query_resource_anchor(query_anchor);
            self.lifetime.query_epoch.check_open(operation).map_err(|_error| self.lifetime.released_error())?;
            return Ok(TenantQueryContext { context });
        }
        let stream_acquire_deadline =
            Instant::now().checked_add(self.service.config.admission_wait_timeout).unwrap_or_else(Instant::now);
        #[cfg(feature = "cugraph")]
        let warm_graph_reservation = require_warm_graph_reservation(
            take_owned_for_cleanup(
                &self.lifetime.warm_graph_reservation,
                "warm graph reservation for context acquisition",
            ),
            self.lifetime.warm_graph_priced,
        )?;
        #[cfg(feature = "cugraph")]
        let (stream_lease, reserved_resident_graph_hit) = match warm_graph_reservation {
            Some(reservation) => {
                (reservation.stream_lease, Some(Arc::new(ReservedResidentGraphHit::new(reservation.preparation_pin))))
            },
            None => {
                let stream_lease = device.stream_pool().acquire_observed_with_cancellation_deadline(
                    &cancellation,
                    stream_acquire_deadline,
                    "nexus_query_engine.query_service.context.stream_acquire",
                    self.lifetime.observation.clone(),
                    stream_scope.clone(),
                )?;
                (stream_lease, None)
            },
        };
        #[cfg(not(feature = "cugraph"))]
        let stream_lease = device.stream_pool().acquire_observed_with_cancellation_deadline(
            &cancellation,
            stream_acquire_deadline,
            "nexus_query_engine.query_service.context.stream_acquire",
            self.lifetime.observation.clone(),
            stream_scope.clone(),
        )?;
        validate_service_stream_pool_device(device.stream_pool(), stream_lease.device())?;
        let stream_lease_guard = Arc::new(stream_lease);
        let memory_grant = self.lifetime.memory_grant()?;
        let runtime = stream_lease_guard.shared_runtime()?;
        let gpu_execution_pool = resources.gpu_execution_pool_for_device(self.device_ordinal)?;
        let grant_memory_domain = create_grant_memory_domain(
            runtime.as_ref(),
            Some(&memory_grant),
            self.service.config.query_memory_domain,
            device.capacity_bytes(),
        )?;
        let grant_reservation_arena = if grant_memory_domain.is_some() {
            None
        } else {
            create_grant_reservation_arena(runtime.as_ref(), Some(&memory_grant))?
        };
        let observation_owner = self.lifetime.observation_owner.clone();
        let query_id = self.lifetime.observation.query_id().map(|query_id| QueryId::new(query_id.as_str()));
        let trace_id =
            self.lifetime.observation.trace_id().map(|trace_id| crate::metrics::TraceId::new(trace_id.as_str()));
        let runtime_observation = self.lifetime.observation.clone();
        let context = ExecutionContext::from_parts(ExecutionContextParts {
            service_epoch: Some(self.service.service_epoch.clone()),
            query_epoch: Some(self.lifetime.query_epoch.clone()),
            query_anchor: None,
            io_lifecycle: None,
            query_id,
            trace_id,
            runtime_observation,
            runtime_observation_owner: observation_owner,
            runtime_execution_scope: None,
            runtime_stream_lease_scope: stream_scope,
            runtime,
            gpu_execution_pool,
            memory_policy: MemoryResourcePolicy::ExternalRuntime,
            memory_control_policy: self.memory_control_policy(),
            memory_grant: Some(memory_grant),
            grant_reservation_arena,
            grant_memory_domain,
            boundary_policy: self.service.config.boundary_policy,
            source_scheduler_policy: self.service.config.source_scheduler_policy,
            source_chunk_planning_mode: self.service.config.source_chunk_planning_mode,
            lakehouse_cache_fill_config: self.service.config.lakehouse_cache_fill.clone(),
            service_stream_pool: device.stream_pool().clone(),
            lakehouse_cache_fill_budget: resources.lakehouse_cache_fill_budget.clone(),
            lakehouse_cache_fill_concurrency: resources.lakehouse_cache_fill_concurrency.clone(),
            lakehouse_cache_fill_events: super::LakehouseCacheFillEventSink::default(),
            object_store_control_runtime: Arc::clone(&resources.object_store_control_runtime),
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
            local_parquet_source_cache: resources.local_parquet_source_cache_for_device(self.device_ordinal),
            #[cfg(feature = "cugraph")]
            resident_graph_cache: resources.resident_graph_cache_for_device(self.device_ordinal),
            #[cfg(feature = "cugraph")]
            reserved_resident_graph_hit,
            stream_lease_guard: Some(stream_lease_guard),
            query_service_admitted: true,
        });
        *root_context = Some(context.clone());
        drop(root_context);
        let context = context.with_query_resource_anchor(query_anchor);
        self.lifetime.query_epoch.check_open(operation).map_err(|_error| self.lifetime.released_error())?;
        Ok(TenantQueryContext { context })
    }

    /// Requests admission release before dropping the handle.
    ///
    /// Runtime-backed services complete GPU resource destruction on the
    /// selected device executor before waking queued admissions.
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
}

impl TenantQueryContext {
    /// Native execution context for this tenant query.
    pub fn context(&self) -> &ExecutionContext {
        &self.context
    }
}

#[cfg(test)]
mod tests {
    use std::{
        panic::{AssertUnwindSafe, catch_unwind},
        sync::{Arc, Mutex},
    };

    #[cfg(feature = "cugraph")]
    use super::require_warm_graph_reservation;
    use super::take_owned_for_cleanup;
    #[cfg(feature = "cugraph")]
    use crate::error::{QueryErrorCode, QueryErrorStatus};

    #[test]
    fn poisoned_owned_resource_is_recovered_for_ordered_cleanup() {
        let owned = Arc::new(Mutex::new(Some(7_u8)));
        let poison = Arc::clone(&owned);
        let _ = catch_unwind(AssertUnwindSafe(move || {
            let _guard = poison.lock().unwrap();
            panic!("poison test resource");
        }));

        assert_eq!(take_owned_for_cleanup(owned.as_ref(), "test resource"), Some(7));
        assert_eq!(take_owned_for_cleanup(owned.as_ref(), "test resource"), None);
    }

    #[cfg(feature = "cugraph")]
    #[test]
    fn consumed_warm_graph_reservation_requires_query_readmission() {
        let error = require_warm_graph_reservation(Option::<()>::None, true).unwrap_err();

        assert_eq!(error.code(), QueryErrorCode::ServiceOverloaded);
        assert_eq!(error.status(), QueryErrorStatus::Temporary);
        assert_eq!(error.fact_value("reason"), Some("stream_and_pin_reservation_already_consumed"));
        assert_eq!(error.fact_value("retry_action"), Some("readmit_query"));
    }
}
