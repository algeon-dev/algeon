use std::sync::Arc;

use super::{
    Error, ExecutionContext, IoLifecycle, NexusNativeExecutionConfig, NexusNativeExecutionContext, QueryAdmissionFacts,
    QueryId, TaskContext, TicketAdmissionContext, TraceId,
};

impl NexusNativeExecutionContext {
    pub(super) fn ticket_bound(ticket_context: TicketAdmissionContext) -> Self {
        let context = ticket_context.context().context().clone();
        let admission_facts = ticket_context.admission_facts();
        Self {
            context,
            admission_facts: Some(admission_facts),
            record_admission_facts: ticket_context.should_record_admission_facts(),
            _admission_context: None,
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
    pub(crate) async fn execution_context_async(
        &self,
        task_context: Arc<TaskContext>,
    ) -> crate::error::Result<NexusNativeExecutionContext> {
        self.validate_query_policies()?;
        if let Some(ticket) = self.query_admission_ticket_for_task_context(&task_context) {
            let task_context =
                super::super::admission::task_context_with_query_admission_ticket(task_context, Arc::clone(&ticket));
            return ticket
                .acquire_context(task_context, self.cancellation.clone())
                .await
                .map(NexusNativeExecutionContext::ticket_bound)
                .and_then(|context| self.stamp_acquired_service_context(context))
                .map_err(|error| {
                    Error::from_native_with_blocking_diagnostics(
                        "datafusion_nexus.native_exec.context",
                        "failed to acquire backend query-service nexus native execution context",
                        self.stamp_native_error_correlation(error).with_fact("stage", "ticket_admission"),
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
        if let Some(lifecycle) = &self.io_lifecycle {
            context = context.with_io_lifecycle(lifecycle.clone());
        }
        Ok(context)
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
