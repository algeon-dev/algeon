//! Acquisition of the native execution context for one fragment.
//!
//! `execution_context_async` is where a fragment admits: it finds the attempt
//! ticket, acquires the attempt's one admission through it, and stamps the
//! context with query and trace correlation. It awaits in the admission queue
//! and returns a typed error when the queue deadline passes, the attempt is
//! cancelled, or no backend is installed; it never creates a context outside
//! an attempt.

use std::sync::Arc;

use query_planning::error::{QueryError, Result as QueryEngineResult};

use super::{
    AlgeonNativeExecutionConfig, AlgeonNativeExecutionContext, ExecutionContext, IoLifecycle, QueryAdmissionFacts,
    QueryId, TaskContext, TicketAdmissionContext, TraceId,
};
use crate::{
    error::{Error, ErrorCode, Result},
    native_exec::admission::task_context_with_query_admission_ticket,
};

impl AlgeonNativeExecutionContext {
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

    pub(super) fn with_query_id(mut self, query_id: QueryId) -> QueryEngineResult<Self> {
        self.context = self.context.with_query_id(query_id)?;
        Ok(self)
    }

    pub(super) fn with_trace_id(mut self, trace_id: TraceId) -> QueryEngineResult<Self> {
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

impl AlgeonNativeExecutionConfig {
    /// Acquires the admitted native context for the fragment executing in
    /// `task_context`.
    ///
    /// Awaits the attempt's admission when this fragment is the first to ask,
    /// or joins the admission already placed. Fails with `InvalidRequest`
    /// (`reason = gpu_backend_required`) when neither the task context nor the
    /// config carries admission authority.
    pub(crate) async fn execution_context_async(
        &self,
        task_context: Arc<TaskContext>,
    ) -> Result<AlgeonNativeExecutionContext> {
        self.validate_query_policies()?;
        if let Some(ticket) = self.query_admission_ticket_for_task_context(&task_context) {
            let task_context = task_context_with_query_admission_ticket(task_context, Arc::clone(&ticket));
            return ticket
                .acquire_context(task_context)
                .await
                .map(AlgeonNativeExecutionContext::ticket_bound)
                .and_then(|context| self.stamp_acquired_service_context(context))
                .map_err(|error| {
                    Error::from_native_with_blocking_diagnostics(
                        "algeon_datafusion.native_exec.context",
                        "failed to acquire backend query-service algeon native execution context",
                        self.stamp_native_error_correlation(error).with_fact("stage", "ticket_admission"),
                    )
                });
        }

        Err(self.gpu_backend_required_error())
    }

    fn gpu_backend_required_error(&self) -> Error {
        let mut error = Error::new(
            ErrorCode::InvalidRequest,
            "algeon_datafusion.native_exec.context",
            "GPU native execution requires admission from an installed AlgeonGpuBackend",
        )
        .with_log_safe_fact("reason", "gpu_backend_required")
        .with_log_safe_fact("remediation", "install_algeon_gpu_backend");
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
        mut context: AlgeonNativeExecutionContext,
    ) -> QueryEngineResult<AlgeonNativeExecutionContext> {
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

    pub(super) fn stamp_native_error_correlation(&self, mut error: QueryError) -> QueryError {
        if let Some(query_id) = &self.query_id {
            error = error.with_query_id(query_id.clone());
        }
        if let Some(trace_id) = &self.trace_id {
            error = error.with_trace_id(trace_id.clone());
        }
        error
    }
}
