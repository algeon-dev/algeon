use datafusion_nexus::native_exec::graph::GraphAdmissionModelAsk;

fn expose_graph_model_admission_evidence(ask: GraphAdmissionModelAsk) {
    let _ = ask.cold_capability_evidence();
    let _ = ask.warm_capability_evidence();
}

fn main() {
    let _ = GraphAdmissionModelAsk::from_graph_memory_admission_evidence;
}
