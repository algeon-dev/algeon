use nexus_query_engine::runtime::{QueryAdmissionRequest, QueryPlacementHint};

fn main() {
    let _ = QueryPlacementHint::new();
    let _ = QueryAdmissionRequest::new().with_placement_hint(Default::default());
}
