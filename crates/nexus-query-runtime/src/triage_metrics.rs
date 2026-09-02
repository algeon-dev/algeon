use crate::{
    metrics::QueryMetrics,
    triage::{TriageCode, TriageEvidenceRef, TriageSeverity, TriageSignal, TriageSummary, TriageVerdict},
};

impl QueryMetrics {
    /// Builds a compact triage summary from runtime metrics.
    pub fn triage_summary(&self) -> TriageSummary {
        let mut signals = Vec::new();
        let query_id = self.query_id.is_assigned().then(|| self.query_id.clone());

        if self.reservation_denied_count > 0 {
            let evidence =
                self.reservation_events.iter().enumerate().find(|(_, event)| event.outcome.as_str() == "denied").map(
                    |(index, event)| {
                        let mut evidence =
                            TriageEvidenceRef::new("query_metrics.rows.memory_reservations").with_row_index(index);
                        if let Some(query_id) = event.query_id.clone().or_else(|| query_id.clone()) {
                            evidence = evidence.with_query_id(query_id);
                        }
                        if let Some(plan_node_id) = event.node_id.clone() {
                            evidence = evidence.with_plan_node_id(plan_node_id);
                        }
                        if let Some(compiled_task_id) = event.compiled_task_id.clone() {
                            evidence = evidence.with_compiled_task_id(compiled_task_id);
                        }
                        evidence
                    },
                );
            signals.push(maybe_with_evidence(
                TriageSignal::new(
                    TriageSeverity::Warning,
                    TriageCode::MemoryReservationDenied,
                    format!("{} memory reservation request(s) were denied", self.reservation_denied_count),
                ),
                evidence,
            ));
        }

        if self.retry_resource_exhausted_count > 0 {
            signals.push(TriageSignal::new(
                TriageSeverity::Warning,
                TriageCode::ResourceExhausted,
                format!("{} retry attempt(s) were caused by resource exhaustion", self.retry_resource_exhausted_count),
            ));
        }
        if self.retry_reservation_failure_count > 0 {
            signals.push(TriageSignal::new(
                TriageSeverity::Warning,
                TriageCode::MemoryReservationDenied,
                format!("{} retry attempt(s) were caused by reservation failure", self.retry_reservation_failure_count),
            ));
        }

        let verdict = if signals.is_empty() { TriageVerdict::Healthy } else { TriageVerdict::Degraded };
        TriageSummary::new(verdict, signals)
    }
}

fn maybe_with_evidence(signal: TriageSignal, evidence: Option<TriageEvidenceRef>) -> TriageSignal {
    match evidence {
        Some(evidence) => signal.with_evidence(evidence),
        None => signal,
    }
}
