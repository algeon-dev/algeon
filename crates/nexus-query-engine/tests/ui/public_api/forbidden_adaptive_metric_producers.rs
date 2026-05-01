use nexus_query_engine::{
    metrics::QueryMetrics,
    triage::TriageCode,
};

fn main() {
    let mut metrics = QueryMetrics::default();
    metrics.record_adaptive_retry();
    metrics.record_adaptive_spill();
    metrics.record_adaptive_downgrade();
    metrics.record_adaptive_memory_history_lookup(true);
    let _ = TriageCode::AdaptivePipelineRejected;
    let _ = TriageCode::AdaptiveRetry;
    let _ = TriageCode::AdaptiveSpill;
    let _ = TriageCode::AdaptiveDowngrade;
}
