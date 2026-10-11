//! The admission authority a config carries.
//!
//! At planning the config holds a `QueryAdmissionSource`, the backend that
//! will admit the plan. Once an invocation minted its ticket, or a composed
//! root bound one, the config can hold that `QueryAdmissionTicket` too. Lookup
//! prefers the ticket of the executing task context over the config's own, so
//! every fragment of one execution admits under the same ticket whatever its
//! config carried at planning.

use std::sync::Arc;

use query_runtime::runtime::{ExecutionCancellation, GpuBackendSession};

use super::{
    AlgeonNativeExecutionConfig, QueryAdmissionSource, QueryAdmissionTicket, TaskContext, invocation_admission_ticket,
};
use crate::{error::Result, native_exec::admission::attempt_execution_required_error, planner::native::Invocation};

impl AlgeonNativeExecutionConfig {
    pub(crate) fn with_query_admission_source(mut self, source: QueryAdmissionSource) -> Self {
        self.query_admission_source = Some(source);
        self.query_admission_ticket = None;
        self
    }

    pub(crate) fn query_admission_source(&self) -> Option<QueryAdmissionSource> {
        self.query_admission_source.clone()
    }

    pub(crate) fn with_existing_query_admission_ticket(mut self, ticket: Arc<QueryAdmissionTicket>) -> Self {
        self.query_admission_ticket = Some(ticket);
        self
    }

    /// Drops a planner-time ticket before a frontend resolves metadata that
    /// participates in its immutable admission evidence.
    #[cfg(feature = "cugraph")]
    pub(crate) fn without_query_admission_ticket(mut self) -> Self {
        self.query_admission_ticket = None;
        self
    }

    #[cfg(test)]
    pub(crate) fn query_admission_ticket_id(&self) -> Option<u64> {
        self.query_admission_ticket.as_ref().map(|ticket| ticket.id())
    }

    pub(crate) fn query_admission_ticket_for_task_context(
        &self,
        task_context: &TaskContext,
    ) -> Option<Arc<QueryAdmissionTicket>> {
        invocation_admission_ticket(task_context).or_else(|| self.query_admission_ticket.as_ref().map(Arc::clone))
    }

    #[cfg(test)]
    pub(crate) fn query_admission_ticket_id_for_task_context(&self, task_context: &TaskContext) -> Option<u64> {
        self.query_admission_ticket_for_task_context(task_context).map(|ticket| ticket.id())
    }

    /// The cancellation of the attempt that native work in `task_context`
    /// runs under: the invocation's attempt token, else the token of the
    /// attempt its admission ticket belongs to.
    pub(crate) fn attempt_cancellation(
        &self,
        task_context: &TaskContext,
        operation: &'static str,
    ) -> Result<ExecutionCancellation> {
        Invocation::of_task(task_context)
            .and_then(Invocation::attempt)
            .map(|attempt| attempt.cancellation().execution_cancellation())
            .or_else(|| {
                self.query_admission_ticket_for_task_context(task_context).map(|ticket| ticket.attempt_cancellation())
            })
            .ok_or_else(|| attempt_execution_required_error(operation))
    }

    pub(crate) fn query_admission_ticket(&self) -> Option<Arc<QueryAdmissionTicket>> {
        self.query_admission_ticket.as_ref().map(Arc::clone)
    }

    pub fn has_query_admission_authority(&self) -> bool {
        self.query_admission_source.is_some() || self.query_admission_ticket.is_some()
    }

    pub(crate) fn query_admission_source_matches(&self, session: &GpuBackendSession) -> bool {
        self.query_admission_source.as_ref().is_some_and(|source| source.backend_matches(session))
    }
}
