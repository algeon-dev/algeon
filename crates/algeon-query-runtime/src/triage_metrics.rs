use query_planning::triage::{TriageCode, TriageSeverity, TriageSignal, TriageSummary, TriageVerdict};

use crate::metrics::QueryMetrics;

impl QueryMetrics {
    /// Builds a compact triage summary from runtime metrics.
    pub fn triage_summary(&self) -> TriageSummary {
        let mut signals = Vec::new();
        if self.retry_resource_exhausted_count > 0 {
            signals.push(TriageSignal::new(
                TriageSeverity::Warning,
                TriageCode::ResourceExhausted,
                format!("{} retry attempt(s) were caused by resource exhaustion", self.retry_resource_exhausted_count),
            ));
        }
        let verdict = if signals.is_empty() { TriageVerdict::Healthy } else { TriageVerdict::Degraded };
        TriageSummary::new(verdict, signals)
    }
}
