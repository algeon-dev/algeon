use nexus_query_engine::{admission::bounded_pipeline_attempt_requirement, runtime::QueryAdmissionRequest};

fn main() {
    let _ = bounded_pipeline_attempt_requirement;
    let _ = QueryAdmissionRequest::with_test_planned_capability_evidence;
}
