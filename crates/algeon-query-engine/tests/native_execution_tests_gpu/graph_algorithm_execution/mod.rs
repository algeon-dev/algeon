use arrow::datatypes::{Float32Type, Float64Type, Int32Type, Int64Type};
use empty_inputs::graph_input_handle;
use query_planning::{error::QueryErrorStatus, plan::graph_result_schema_for_vertex_type};
use support::{
    gpu_resident_source, graph_spec_for_vertex_type, graph_vertex_values, int32_graph_spec,
    int32_stable_graph_result_schema, stable_graph_result_schema_for_vertex_type,
};

use super::*;

mod betweenness;
mod bfs;
mod cache;
mod empty_inputs;
mod reproducible_results;
mod similarity;
mod support;

#[test]
fn graph_algorithm_execution_cases_1() {
    run_named_cases(&[
        (
            "graph_algorithm_execution::betweenness::betweenness_k_resolves_to_sorted_first_vertices",
            betweenness::betweenness_k_resolves_to_sorted_first_vertices,
        ),
        (
            "graph_algorithm_execution::reproducible_results::scc_labels_are_min_vertex_ids_and_stable_across_runs",
            reproducible_results::scc_labels_are_min_vertex_ids_and_stable_across_runs,
        ),
        (
            "graph_algorithm_execution::similarity::pairwise_similarity_executes_all_metrics_with_stable_schema_and_multiset_pairs",
            similarity::pairwise_similarity_executes_all_metrics_with_stable_schema_and_multiset_pairs,
        ),
        (
            "graph_algorithm_execution::betweenness::betweenness_k_selects_deterministic_approximate_seed_count",
            betweenness::betweenness_k_selects_deterministic_approximate_seed_count,
        ),
        (
            "graph_algorithm_execution::cache::source_alias_residency_does_not_preserve_graph_residency_descriptor",
            cache::source_alias_residency_does_not_preserve_graph_residency_descriptor,
        ),
        (
            "graph_algorithm_execution::betweenness::exact_betweenness_threshold_uses_actual_vertices_not_edge_upper_bound",
            betweenness::exact_betweenness_threshold_uses_actual_vertices_not_edge_upper_bound,
        ),
        (
            "graph_algorithm_execution::similarity::pairwise_similarity_allows_one_pair_column_to_supply_both_self_pair_endpoints",
            similarity::pairwise_similarity_allows_one_pair_column_to_supply_both_self_pair_endpoints,
        ),
        (
            "graph_algorithm_execution::bfs::bfs_keeps_null_weight_edges_but_drops_null_edge_id_rows",
            bfs::bfs_keeps_null_weight_edges_but_drops_null_edge_id_rows,
        ),
        (
            "graph_algorithm_execution::similarity::pairwise_similarity_weighted_jaccard_uses_the_graph_weight_binding",
            similarity::pairwise_similarity_weighted_jaccard_uses_the_graph_weight_binding,
        ),
        (
            "graph_algorithm_execution::reproducible_results::wcc_labels_are_min_vertex_ids_and_stable_across_runs",
            reproducible_results::wcc_labels_are_min_vertex_ids_and_stable_across_runs,
        ),
    ]);
}

#[test]
fn graph_algorithm_execution_cases_2() {
    run_named_cases(&[
        (
            "graph_algorithm_execution::bfs::bfs_drops_null_edge_rows_before_graph_construction",
            bfs::bfs_drops_null_edge_rows_before_graph_construction,
        ),
        (
            "graph_algorithm_execution::reproducible_results::betweenness_scores_are_bitwise_stable_across_device_budgets",
            reproducible_results::betweenness_scores_are_bitwise_stable_across_device_budgets,
        ),
        (
            "graph_algorithm_execution::similarity::bounded_similarity_accepts_gpu_pair_input_exactly_once",
            similarity::bounded_similarity_accepts_gpu_pair_input_exactly_once,
        ),
        (
            "graph_algorithm_execution::bfs::bfs_target_predicate_keeps_raw_output_and_target_diagnostics",
            bfs::bfs_target_predicate_keeps_raw_output_and_target_diagnostics,
        ),
        (
            "graph_algorithm_execution::reproducible_results::betweenness_scores_are_bitwise_stable_across_runs",
            reproducible_results::betweenness_scores_are_bitwise_stable_across_runs,
        ),
        (
            "graph_algorithm_execution::betweenness::betweenness_with_explicit_seeds_records_approximate_metadata",
            betweenness::betweenness_with_explicit_seeds_records_approximate_metadata,
        ),
        (
            "graph_algorithm_execution::bfs::bfs_null_weight_does_not_drop_reachable_edges",
            bfs::bfs_null_weight_does_not_drop_reachable_edges,
        ),
        (
            "graph_algorithm_execution::empty_inputs::bfs_empty_edges_with_nonempty_vertex_predicate_is_a_graph_input_error",
            empty_inputs::bfs_empty_edges_with_nonempty_vertex_predicate_is_a_graph_input_error,
        ),
        (
            "graph_algorithm_execution::similarity::pairwise_similarity_rejects_null_pairs_absent_endpoints_and_pair_graph_dtype_mismatch",
            similarity::pairwise_similarity_rejects_null_pairs_absent_endpoints_and_pair_graph_dtype_mismatch,
        ),
        (
            "graph_algorithm_execution::similarity::pairwise_similarity_accepts_int64_pairs_and_short_circuits_empty_pairs",
            similarity::pairwise_similarity_accepts_int64_pairs_and_short_circuits_empty_pairs,
        ),
    ]);
}

#[test]
fn graph_algorithm_execution_cases_3() {
    run_named_cases(&[
        (
            "graph_algorithm_execution::bfs::bfs_empty_path_endpoint_policy_returns_empty_after_edge_filter",
            bfs::bfs_empty_path_endpoint_policy_returns_empty_after_edge_filter,
        ),
        ("graph_algorithm_execution::bfs::bfs_include_edge_ids_filters_edges", bfs::bfs_include_edge_ids_filters_edges),
        (
            "graph_algorithm_execution::bfs::bfs_source_vertices_side_input_accepts_multiple_rows",
            bfs::bfs_source_vertices_side_input_accepts_multiple_rows,
        ),
        (
            "graph_algorithm_execution::empty_inputs::pagerank_empty_edges_returns_a_schema_correct_empty_result_after_null_sanitization",
            empty_inputs::pagerank_empty_edges_returns_a_schema_correct_empty_result_after_null_sanitization,
        ),
        (
            "graph_algorithm_execution::bfs::bfs_include_edges_relation_filters_by_source_destination",
            bfs::bfs_include_edges_relation_filters_by_source_destination,
        ),
        (
            "graph_algorithm_execution::betweenness::bounded_exact_betweenness_above_threshold_is_refused_before_graph_memory_denial",
            betweenness::bounded_exact_betweenness_above_threshold_is_refused_before_graph_memory_denial,
        ),
        (
            "graph_algorithm_execution::betweenness::exact_betweenness_above_threshold_is_refused_before_cugraph",
            betweenness::exact_betweenness_above_threshold_is_refused_before_cugraph,
        ),
        (
            "graph_algorithm_execution::empty_inputs::personalized_pagerank_rejects_an_empty_required_side_input_before_graph_execution",
            empty_inputs::personalized_pagerank_rejects_an_empty_required_side_input_before_graph_execution,
        ),
        (
            "graph_algorithm_execution::empty_inputs::bfs_strict_rejects_empty_edges_as_a_permanent_graph_input_error",
            empty_inputs::bfs_strict_rejects_empty_edges_as_a_permanent_graph_input_error,
        ),
        (
            "graph_algorithm_execution::bfs::bfs_source_vertices_side_input_rejects_empty_source_set",
            bfs::bfs_source_vertices_side_input_rejects_empty_source_set,
        ),
    ]);
}
