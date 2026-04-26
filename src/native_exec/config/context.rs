use super::*;

impl NexusNativeExecutionContext {
    pub(super) fn standalone(context: ExecutionContext, task_context_owner: Option<usize>) -> Self {
        #[cfg(debug_assertions)]
        let standalone_bounded_context_guard = standalone_bounded_context_debug_guard(&context, task_context_owner);
        #[cfg(not(debug_assertions))]
        let _ = task_context_owner;
        Self {
            context,
            admission_facts: None,
            record_admission_facts: false,
            _tenant_context: None,
            ticket_context: None,
            #[cfg(debug_assertions)]
            _standalone_bounded_context_guard: standalone_bounded_context_guard,
        }
    }

    pub(super) fn tenant_bound(
        tenant_context: TenantQueryContext,
        admission_facts: QueryAdmissionFacts,
        record_admission_facts: bool,
    ) -> Self {
        let context = tenant_context.context().clone();
        Self {
            context,
            admission_facts: Some(admission_facts),
            record_admission_facts,
            _tenant_context: Some(tenant_context),
            ticket_context: None,
            #[cfg(debug_assertions)]
            _standalone_bounded_context_guard: None,
        }
    }

    pub(super) fn ticket_bound(ticket_context: TicketTenantQueryContext) -> Self {
        let context = ticket_context.context().context().clone();
        let admission_facts = ticket_context.admission_facts();
        Self {
            context,
            admission_facts: Some(admission_facts),
            record_admission_facts: ticket_context.should_record_admission_facts(),
            _tenant_context: None,
            ticket_context: Some(ticket_context),
            #[cfg(debug_assertions)]
            _standalone_bounded_context_guard: None,
        }
    }

    pub(super) fn with_query_id(mut self, query_id: QueryId) -> nexus_query_engine::error::Result<Self> {
        self.context = self.context.with_query_id(query_id)?;
        Ok(self)
    }

    pub(super) fn with_trace_id(mut self, trace_id: TraceId) -> nexus_query_engine::error::Result<Self> {
        self.context = self.context.with_trace_id(trace_id)?;
        Ok(self)
    }

    pub(super) fn with_graph_memory_calibration_override(mut self, override_: GraphMemoryCalibrationOverride) -> Self {
        self.context = self.context.with_graph_memory_calibration_override(override_);
        self
    }

    pub(super) fn with_io_lifecycle(mut self, lifecycle: IoLifecycle) -> Self {
        self.context = self.context.clone().with_io_lifecycle(lifecycle);
        self
    }

    pub(crate) fn context(&self) -> &ExecutionContext {
        &self.context
    }

    pub(crate) fn task_context(&self) -> Option<&Arc<TaskContext>> {
        self.ticket_context.as_ref().map(|ticket_context| ticket_context.task_context())
    }

    pub(crate) fn admission_facts(&self) -> Option<QueryAdmissionFacts> {
        self.record_admission_facts.then_some(())?;
        self.admission_facts
    }
}

#[cfg(debug_assertions)]
#[derive(Debug)]
pub(in crate::native_exec) struct StandaloneBoundedContextDebugGuard {
    pub(super) device_ordinal: usize,
    pub(super) task_context_owner: Option<usize>,
}

#[cfg(debug_assertions)]
pub(super) type StandaloneBoundedContextCounts =
    std::sync::Mutex<std::collections::BTreeMap<usize, std::collections::BTreeMap<Option<usize>, usize>>>;

#[cfg(debug_assertions)]
pub(super) fn standalone_bounded_context_counts() -> &'static StandaloneBoundedContextCounts {
    static COUNTS: OnceLock<StandaloneBoundedContextCounts> = OnceLock::new();
    COUNTS.get_or_init(Default::default)
}

#[cfg(debug_assertions)]
pub(super) fn standalone_bounded_context_debug_guard(
    context: &ExecutionContext,
    task_context_owner: Option<usize>,
) -> Option<StandaloneBoundedContextDebugGuard> {
    if !matches!(context.memory_control_policy(), MemoryControlPolicy::Bounded { .. }) {
        return None;
    }
    let device_ordinal = match context.device_placement() {
        nexus_query_engine::runtime::DevicePlacement::SingleDevice { device_ordinal } => device_ordinal,
        _ => return None,
    };
    Some(StandaloneBoundedContextDebugGuard::new_with_owner(device_ordinal, task_context_owner))
}

#[cfg(debug_assertions)]
impl StandaloneBoundedContextDebugGuard {
    #[cfg(test)]
    pub(super) fn new(device_ordinal: usize) -> Self {
        Self::new_with_owner(device_ordinal, None)
    }

    pub(super) fn new_with_owner(device_ordinal: usize, task_context_owner: Option<usize>) -> Self {
        let guard = Self { device_ordinal, task_context_owner };
        let has_conflicting_owner = {
            let mut counts = standalone_bounded_context_counts()
                .lock()
                .expect("standalone bounded native context debug registry poisoned");
            let device_counts = counts.entry(device_ordinal).or_default();
            let has_conflicting_owner = !device_counts.is_empty();
            *device_counts.entry(task_context_owner).or_default() += 1;
            has_conflicting_owner
        };
        debug_assert!(
            !has_conflicting_owner,
            "concurrent unadmitted bounded native contexts on CUDA device {device_ordinal}; use NexusGpuBackend to share QueryService admission"
        );
        guard
    }
}

#[cfg(debug_assertions)]
impl Drop for StandaloneBoundedContextDebugGuard {
    fn drop(&mut self) {
        if let Ok(mut counts) = standalone_bounded_context_counts().lock()
            && let Some(device_counts) = counts.get_mut(&self.device_ordinal)
        {
            if let Some(count) = device_counts.get_mut(&self.task_context_owner) {
                *count = count.saturating_sub(1);
                if *count == 0 {
                    device_counts.remove(&self.task_context_owner);
                }
            }
            if device_counts.is_empty() {
                counts.remove(&self.device_ordinal);
            }
        }
    }
}

impl NexusNativeExecutionConfig {
    #[cfg(test)]
    pub(crate) fn execution_context(&self) -> crate::error::Result<NexusNativeExecutionContext> {
        self.execution_context_with_standalone_owner(None)
    }

    pub(super) fn execution_context_with_standalone_owner(
        &self,
        standalone_owner: Option<usize>,
    ) -> crate::error::Result<NexusNativeExecutionContext> {
        self.validate_query_policies()?;
        if let Some(handle) = &self.query_service_handle {
            let admission_facts = QueryAdmissionFacts::from_handle(handle.as_ref());
            let admission_facts_recorded = self.query_admission_facts_recorded.clone();
            return handle
                .acquire_context_with_cancellation(self.cancellation.clone())
                .map(|tenant_context| {
                    let record_admission_facts = admission_facts_recorded
                        .as_ref()
                        .is_some_and(|recorded| !recorded.swap(true, Ordering::AcqRel));
                    NexusNativeExecutionContext::tenant_bound(tenant_context, admission_facts, record_admission_facts)
                })
                .and_then(|context| self.stamp_acquired_service_context(context))
                .map_err(|error| {
                    Error::from_native(
                        "datafusion_nexus.native_exec.context",
                        "failed to acquire query service nexus native execution context",
                        self.stamp_native_error_correlation(error),
                    )
                });
        }

        let mut builder = ExecutionContext::builder()
            .with_memory_policy(self.memory_resource_policy)
            .with_memory_control_policy(self.memory_control_policy)
            .with_boundary_policy(self.boundary_policy)
            .with_source_scheduler_policy(self.source_scheduler_policy)
            .with_source_chunk_planning_mode(self.source_chunk_planning_mode)
            .with_lakehouse_cache_fill_config(self.lakehouse_cache_fill_config.clone())
            .with_cancellation(self.cancellation.clone())
            .with_native_pipeline_execution_mode(self.native_pipeline_execution_mode)
            .with_statistical_aggregate_execution_mode(self.stddev_samp_execution_mode)
            .with_aggregate_strategy_execution_options(self.aggregate_strategy_execution_options)
            .with_source_chunk_planning_options(self.source_chunk_planning_options)
            .with_graph_memory_calibration_override(self.graph_memory_calibration_override)
            .with_source_storage_profiles(self.source_storage_profiles.clone());
        if let Some(lifecycle) = &self.io_lifecycle {
            builder = builder.with_io_lifecycle(lifecycle.clone());
        }
        if let Some(pool) = &self.service_stream_pool {
            builder = builder.with_service_stream_pool(pool.clone());
        }
        if let Some(pool_size) = self.service_stream_pool_size {
            builder = builder.with_service_stream_pool_size(pool_size);
        }
        if let Some(runtime) = &self.runtime {
            builder = builder.with_shared_runtime(Arc::clone(runtime));
        }
        if let Some(device_ordinal) = self.device_ordinal {
            builder = builder.with_device_ordinal(device_ordinal);
        }
        if let Some(cache) = &self.hash_join_build_cache {
            builder = builder.with_hash_join_build_cache(Arc::clone(cache));
        }
        if let Some(cache) = &self.local_parquet_source_cache {
            builder = builder.with_local_parquet_source_cache(Arc::clone(cache));
        }
        if let Some(memory_history) = &self.adaptive_memory_history {
            builder = builder.with_adaptive_memory_history(Arc::clone(memory_history));
        }
        if let Some(query_id) = &self.query_id {
            builder = builder.with_query_id(query_id.clone());
        }
        if let Some(trace_id) = &self.trace_id {
            builder = builder.with_trace_id(trace_id.clone());
        }
        builder.build().map(|context| NexusNativeExecutionContext::standalone(context, standalone_owner)).map_err(
            |error| {
                Error::from_native(
                    "datafusion_nexus.native_exec.context",
                    "failed to build nexus native execution context",
                    self.stamp_native_error_correlation(error),
                )
            },
        )
    }

    pub(crate) async fn execution_context_async(
        &self,
        task_context: Arc<TaskContext>,
    ) -> crate::error::Result<NexusNativeExecutionContext> {
        self.validate_query_policies()?;
        if let Some(ticket) = self.query_admission_ticket_for_task_context(&task_context) {
            return ticket
                .acquire_context(task_context, self.cancellation.clone())
                .await
                .map(NexusNativeExecutionContext::ticket_bound)
                .and_then(|context| self.stamp_acquired_service_context(context))
                .map_err(|error| {
                    Error::from_native(
                        "datafusion_nexus.native_exec.context",
                        "failed to acquire backend query-service nexus native execution context",
                        self.stamp_native_error_correlation(error),
                    )
                });
        }
        if let Some(handle) = &self.query_service_handle {
            let admission_facts = QueryAdmissionFacts::from_handle(handle.as_ref());
            let admission_facts_recorded = self.query_admission_facts_recorded.clone();
            return Arc::clone(handle)
                .acquire_context_with_cancellation_async(self.cancellation.clone())
                .await
                .map(|tenant_context| {
                    let record_admission_facts = admission_facts_recorded
                        .as_ref()
                        .is_some_and(|recorded| !recorded.swap(true, Ordering::AcqRel));
                    NexusNativeExecutionContext::tenant_bound(tenant_context, admission_facts, record_admission_facts)
                })
                .and_then(|context| self.stamp_acquired_service_context(context))
                .map_err(|error| {
                    Error::from_native(
                        "datafusion_nexus.native_exec.context",
                        "failed to acquire query service nexus native execution context",
                        self.stamp_native_error_correlation(error),
                    )
                });
        }

        let pool = if tokio::runtime::Handle::try_current().is_ok() {
            let pool_config = self.clone();
            tokio::task::spawn_blocking(move || pool_config.standalone_gpu_execution_pool()).await.map_err(
                |error| {
                    let (code, reason) = if error.is_cancelled() {
                        (crate::error::ErrorCode::Cancelled, "discovery_task_cancelled")
                    } else {
                        (crate::error::ErrorCode::Internal, "discovery_task_panicked")
                    };
                    Error::new(
                        code,
                        "datafusion_nexus.native_exec.context_executor",
                        "standalone GPU execution-pool discovery task failed",
                    )
                    .with_fact("reason", reason)
                    .with_source(error)
                },
            )??
        } else {
            self.standalone_gpu_execution_pool()?
        };
        let config = self.clone();
        let standalone_owner = Some(Arc::as_ptr(&task_context) as usize);
        pool.run(GpuExecutionJobKind::ContextSetup, async move {
            config.execution_context_with_standalone_owner(standalone_owner)
        })
        .await
        .map_err(|error| {
            Error::from_native(
                "datafusion_nexus.native_exec.context_executor",
                "failed to run nexus native context setup on the GPU execution pool",
                self.stamp_native_error_correlation(error),
            )
        })?
    }

    pub(super) fn standalone_gpu_execution_pool(&self) -> crate::error::Result<GpuExecutionPool> {
        let device = match &self.runtime {
            Some(runtime) => runtime.device(),
            None => cudf::DeviceId::new(self.device_ordinal.unwrap_or(cudf::DeviceId::DEFAULT.ordinal())).map_err(
                |error| {
                    Error::from_native(
                        "datafusion_nexus.native_exec.context_executor",
                        "failed to resolve the standalone GPU execution-pool device",
                        nexus_query_engine::error::QueryError::from_cudf(
                            "datafusion_nexus.native_exec.context_executor",
                            "failed to resolve the standalone GPU execution-pool device",
                            error,
                        ),
                    )
                },
            )?,
        };
        let worker_count = self
            .service_stream_pool
            .as_ref()
            .map(ServiceStreamPool::pool_size)
            .or(self.service_stream_pool_size)
            .unwrap_or_else(|| default_service_stream_pool_size_for_device(device));
        GpuExecutionPool::shared_for_device(device.ordinal(), worker_count).map_err(|error| {
            Error::from_native(
                "datafusion_nexus.native_exec.context_executor",
                "failed to resolve the standalone GPU execution pool",
                self.stamp_native_error_correlation(error),
            )
        })
    }

    pub(super) fn stamp_acquired_service_context(
        &self,
        mut context: NexusNativeExecutionContext,
    ) -> nexus_query_engine::error::Result<NexusNativeExecutionContext> {
        // An ambient composed-query ticket owns the canonical identity. Child
        // fragment configs may carry independently minted planning ids and
        // must not overwrite the pre-admission observation identity.
        if context.context().query_id().is_none()
            && let Some(query_id) = self.query_id.clone()
        {
            context = context.with_query_id(query_id)?;
        }
        if context.context().trace_id().is_none()
            && let Some(trace_id) = self.trace_id.clone()
        {
            context = context.with_trace_id(trace_id)?;
        }
        let current = context.context().graph_memory_calibration_override();
        let graph_memory_calibration_override =
            self.graph_memory_calibration_override_for_acquired_service_context(current);
        if graph_memory_calibration_override != current {
            context = context.with_graph_memory_calibration_override(graph_memory_calibration_override);
        }
        if let Some(lifecycle) = &self.io_lifecycle {
            context = context.with_io_lifecycle(lifecycle.clone());
        }
        Ok(context)
    }

    pub(crate) fn graph_memory_calibration_override_for_acquired_service_context(
        &self,
        mut current: GraphMemoryCalibrationOverride,
    ) -> GraphMemoryCalibrationOverride {
        if self.graph_raw_input_edges_override_configured {
            current.raw_input_edges = self.graph_memory_calibration_override.raw_input_edges;
        }
        current
    }

    pub(super) fn stamp_native_error_correlation(
        &self,
        mut error: nexus_query_engine::error::QueryError,
    ) -> nexus_query_engine::error::QueryError {
        if let Some(query_id) = &self.query_id {
            error = error.with_query_id(query_id.clone());
        }
        if let Some(trace_id) = &self.trace_id {
            error = error.with_trace_id(trace_id.clone());
        }
        error
    }
}
