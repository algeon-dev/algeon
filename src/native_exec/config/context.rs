use super::*;

impl NexusNativeExecutionContext {
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

impl NexusNativeExecutionConfig {
    #[cfg(test)]
    pub(crate) fn execution_context(&self) -> crate::error::Result<NexusNativeExecutionContext> {
        self.validate_query_policies()?;
        if let Some(handle) = &self.query_service_handle {
            let admission_facts = QueryAdmissionFacts::from_query_handle(handle.as_ref());
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
        Err(self.gpu_backend_required_error())
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
            let admission_facts = QueryAdmissionFacts::from_query_handle(handle.as_ref());
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

        Err(self.gpu_backend_required_error())
    }

    fn gpu_backend_required_error(&self) -> Error {
        let mut error = Error::new(
            crate::error::ErrorCode::InvalidRequest,
            "datafusion_nexus.native_exec.context",
            "GPU native execution requires admission from an installed NexusGpuBackend",
        )
        .with_log_safe_fact("reason", "gpu_backend_required")
        .with_log_safe_fact("remediation", "install_nexus_gpu_backend");
        if let Some(query_id) = &self.query_id {
            error = error.with_fact("query_id", query_id.as_str());
        }
        if let Some(trace_id) = &self.trace_id {
            error = error.with_fact("trace_id", trace_id.as_str());
        }
        error
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
