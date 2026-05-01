use nexus_query_engine::exec::{
    force_next_graph_adaptive_chunk_attempt_resource_exhausted_for_test,
    force_next_graph_adaptive_full_attempt_resource_exhausted_for_test,
    force_next_graph_seed_chunk_retry_chunk_attempt_resource_exhausted_for_test,
    force_next_graph_seed_chunk_retry_full_attempt_resource_exhausted_for_test,
    force_next_graph_statistics_optional_resource_exhausted_for_test,
};

fn main() {
    force_next_graph_adaptive_full_attempt_resource_exhausted_for_test();
    force_next_graph_adaptive_chunk_attempt_resource_exhausted_for_test();
    force_next_graph_seed_chunk_retry_full_attempt_resource_exhausted_for_test();
    force_next_graph_seed_chunk_retry_chunk_attempt_resource_exhausted_for_test();
    force_next_graph_statistics_optional_resource_exhausted_for_test();
}
