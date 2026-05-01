use nexus_query_engine::admission::{AttemptCapabilityEvidence, EstimateQuality};

fn main() {
    let _ = AttemptCapabilityEvidence {
        configured_floor_bytes: 1,
        capability_floor_raise_bytes: 1,
        soft_working_set_hint_bytes: Some(1),
        estimate_quality: EstimateQuality::Measured,
        required_capabilities: Vec::new(),
    };
}
