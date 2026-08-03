use nexus_query_engine::{
    admission::AttemptCapabilityEvidence,
    graph_memory::{GraphMemoryAdmissionEvidence, GraphMemoryEstimate},
    runtime::QueryAdmissionRequest,
};

fn derive_admission_evidence_from_a_report_value(estimate: &GraphMemoryEstimate) {
    let _ = AttemptCapabilityEvidence::graph_execution_with_cold_memory_estimate(estimate);
    let _ = AttemptCapabilityEvidence::graph_execution_with_warm_memory_estimate(estimate);
    let _ = GraphMemoryAdmissionEvidence::from_modeled_estimate(estimate);
}

fn attach_model_evidence_to_a_public_request(evidence: &GraphMemoryAdmissionEvidence) {
    let _ = QueryAdmissionRequest::new().with_planned_capability_evidence(evidence.cold_capability_evidence());
}

fn main() {}
