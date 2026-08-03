use nexus_query_engine::graph_memory::{
    GraphMemoryModel, GraphPairCardinalitySource, strict_similarity_pair_cardinality_for_plan,
};

fn main() {
    let _ = GraphMemoryModel::estimate_for_adaptive_bounded;
    let _ = GraphMemoryModel::estimate_for_bounded;
    let _ = GraphPairCardinalitySource::is_strict_bounded_evidence;
    let _ = strict_similarity_pair_cardinality_for_plan;
}
