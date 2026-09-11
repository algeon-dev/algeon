use std::sync::Arc;

use nexus_query_engine::admission::AttemptCapabilityEvidence;

use super::{
    NativeExecutionMetricsCapture, NativeMetricsCapture, NexusNativeExecutionConfig, QueryAdmissionSource,
    QueryAdmissionTicket, QueryId, RuntimeObservationConfig, RuntimeObservationOwner, RuntimeTraceId, TaskContext,
    query_admission_ticket_from_task_context,
};
use crate::attempt::NexusAttemptExtension;

impl NexusNativeExecutionConfig {
    pub(crate) fn with_query_admission_source(mut self, source: QueryAdmissionSource) -> Self {
        self.query_admission_source = Some(source);
        self.query_admission_ticket = None;
        self
    }

    pub(crate) fn new_plan_query_admission_ticket_with_fragment_requirements(
        &self,
        fragment_evidence_items: Vec<AttemptCapabilityEvidence>,
    ) -> nexus_query_engine::error::Result<Option<Arc<QueryAdmissionTicket>>> {
        let Some(source) = self.query_admission_source.clone() else {
            return Ok(None);
        };
        if fragment_evidence_items.is_empty() {
            return Ok(None);
        }
        let attempt_capability_evidence = AttemptCapabilityEvidence::aggregate(fragment_evidence_items)?;
        let observation_capture = self.metrics_capture.as_ref().map(NativeMetricsCapture::new_execution);
        let observation = self.new_runtime_observation(observation_capture.as_ref());
        Ok(Some(QueryAdmissionTicket::new_with_capability_evidence_and_observation(
            source,
            Some(attempt_capability_evidence),
            observation,
            observation_capture,
        )))
    }

    pub(super) fn new_runtime_observation(
        &self,
        capture: Option<&NativeExecutionMetricsCapture>,
    ) -> crate::native_exec::NativeRuntimeObservation {
        let query_id = self.query_id.clone().unwrap_or_else(QueryId::mint);
        let trace_id = self.trace_id.as_ref().map(|trace_id| RuntimeTraceId::new(trace_id.as_str()));
        match capture {
            Some(capture) => capture.new_runtime_observation(query_id, trace_id),
            None => {
                RuntimeObservationOwner::new(query_id, trace_id, RuntimeObservationConfig::native_execution()).into()
            },
        }
    }

    #[cfg(feature = "cugraph")]
    pub(crate) fn with_graph_aware_plan_query_admission_ticket(
        mut self,
        fragment_evidence_items: Vec<AttemptCapabilityEvidence>,
    ) -> nexus_query_engine::error::Result<Self> {
        if let Some(ticket) =
            self.new_plan_query_admission_ticket_with_fragment_requirements(fragment_evidence_items)?
        {
            self.query_admission_ticket = Some(ticket);
        }
        Ok(self)
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
        query_admission_ticket_from_task_context(task_context)
            .or_else(|| {
                task_context
                    .session_config()
                    .options()
                    .extensions
                    .get::<NexusAttemptExtension>()
                    .and_then(NexusAttemptExtension::bound_ticket)
            })
            .or_else(|| self.query_admission_ticket.as_ref().map(Arc::clone))
    }

    #[cfg(test)]
    pub(crate) fn query_admission_ticket_id_for_task_context(&self, task_context: &TaskContext) -> Option<u64> {
        self.query_admission_ticket_for_task_context(task_context).map(|ticket| ticket.id())
    }

    pub(crate) fn query_admission_ticket(&self) -> Option<Arc<QueryAdmissionTicket>> {
        self.query_admission_ticket.as_ref().map(Arc::clone)
    }

    pub fn has_query_admission_authority(&self) -> bool {
        self.query_admission_source.is_some() || self.query_admission_ticket.is_some()
    }

    pub(crate) fn query_admission_source_matches(
        &self,
        session: &nexus_query_engine::runtime::GpuBackendSession,
    ) -> bool {
        self.query_admission_source.as_ref().is_some_and(|source| source.backend_matches(session))
    }
}
