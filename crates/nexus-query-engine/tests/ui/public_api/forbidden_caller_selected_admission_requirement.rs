use nexus_query_engine::{
    admission::{AttemptCapabilityEvidence, EstimateQuality},
    runtime::QueryAdmissionRequest,
};

mod removed_requirement_type {
    use nexus_query_engine::admission::AttemptRequirement;

    #[allow(dead_code)]
    fn accepts_caller_requirement(_: AttemptRequirement) {}
}

fn raw_evidence_fields_must_remain_private() {
    let _ = AttemptCapabilityEvidence {
        configured_floor_bytes: 1,
        capability_floor_raise_bytes: 1,
        soft_working_set_hint_bytes: Some(1),
        estimate_quality: EstimateQuality::Measured,
        required_capabilities: Vec::new(),
    };
}

fn raw_requirement_attachment_must_remain_private() {
    let _ = QueryAdmissionRequest::new().with_attempt_requirement(AttemptCapabilityEvidence::graph_execution());
}

fn main() {
    raw_evidence_fields_must_remain_private();
    raw_requirement_attachment_must_remain_private();
}
