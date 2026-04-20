use std::{collections::BTreeSet, future::Future};

use datafusion::{common::config::ConfigOptions, physical_optimizer::PhysicalOptimizerRule};
use datafusion_nexus::{
    native::lower_physical_plan_to_native,
    native_report::{
        NativeDataFusionPlanningOutcome, NativeDataFusionPlanningReport, NativeDataFusionPostNativeExecutionPath,
        NativeDataFusionRuntimeOutcome, NativeTpcdsParityPlanningOutcome, NativeTpcdsParityReport,
        ParityComparisonMode, ParityFloatTolerance, ParityStageStatus, ParityStatus,
    },
    planner::CudfNativePhysicalOptimizerRule,
};
use datafusion_nexus_bench::{
    tpcds_datafusion_planning_workload_items_with_optimizer_config, tpcds_native_report_session_config,
    tpcds_report_session_config,
};
use nexus_query_engine::mode::{NativePipelineExecutionMode, StatisticalAggregateExecutionMode};

use super::support::*;

const TPCDS_SF1_HEAVY_PLANNING_STACK_SIZE: usize = 32 * 1024 * 1024;

fn run_tpcds_sf1_heavy_planning_test<F, Fut>(name: &'static str, test: F)
where
    F: FnOnce() -> Fut + Send + 'static,
    Fut: Future<Output = ()> + 'static,
{
    std::thread::Builder::new()
        .name(name.to_string())
        .stack_size(TPCDS_SF1_HEAVY_PLANNING_STACK_SIZE)
        .spawn(|| {
            tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap().block_on(test());
        })
        .unwrap()
        .join()
        .unwrap();
}

async fn assert_tpcds_sf1_single_query_whole_native_parity(
    root: &std::path::Path,
    query_id: &str,
    comparison_mode: ParityComparisonMode,
    output_rows: usize,
) {
    let query_filter = vec![query_id.to_string()];
    let items = tpcds_parity_workload_items(
        root,
        sf1_tpcds_test_session_config(),
        sf1_native_optimizer_config(),
        Some(&query_filter),
    )
    .await
    .unwrap();
    let report = NativeTpcdsParityReport::collect(items, ParityFloatTolerance::exact());
    assert_eq!(report.rows.len(), 1);
    let row = &report.rows[0];
    assert_whole_native_parity_row(row, query_id, comparison_mode, output_rows);
}

async fn assert_tpcds_sf1_single_query_partial_native_parity(
    root: &std::path::Path,
    query_id: &str,
    comparison_mode: ParityComparisonMode,
    output_rows: usize,
    native_plan_fragment_count: usize,
    host_datafusion_boundary_count: usize,
) {
    let query_filter = vec![query_id.to_string()];
    let items = tpcds_parity_workload_items(
        root,
        sf1_tpcds_test_session_config(),
        sf1_native_optimizer_config(),
        Some(&query_filter),
    )
    .await
    .unwrap();
    let report = NativeTpcdsParityReport::collect(items, ParityFloatTolerance::exact());
    assert_eq!(report.rows.len(), 1);
    let row = &report.rows[0];
    assert_partial_native_parity_row(
        row,
        query_id,
        comparison_mode,
        output_rows,
        native_plan_fragment_count,
        host_datafusion_boundary_count,
    );
}

#[tokio::test]
async fn tpcds_sf1_q06_matches_with_string_key_hash_prebuild_fragments_on_cpu() {
    let Some(root) = sf1_fixture_root_or_skip() else {
        return;
    };
    let root = root.as_path();
    assert!(root.join("store_sales.parquet").is_file(), "TPC-DS SF1 fixture is missing at {}", root.display());

    let query_filter = vec!["q06".to_string()];
    let items = tpcds_parity_workload_items(
        root,
        sf1_tpcds_test_session_config(),
        sf1_native_optimizer_config(),
        Some(&query_filter),
    )
    .await
    .unwrap();
    let report = NativeTpcdsParityReport::collect(items, ParityFloatTolerance::exact());
    assert_eq!(report.rows.len(), 1);

    let row = &report.rows[0];
    assert_partial_native_parity_row(row, "q06", ParityComparisonMode::OrderedExact, 45, 3, 7);
    assert_ne!(row.parity_status, ParityStatus::NativeRejected);
}

#[tokio::test]
async fn tpcds_sf1_q85_matches_after_shared_resident_pipeline_contract_fix() {
    let Some(root) = sf1_fixture_root_or_skip() else {
        return;
    };
    let root = root.as_path();
    assert!(root.join("web_sales.parquet").is_file(), "TPC-DS SF1 fixture is missing at {}", root.display());

    let plain_plan = plain_tpcds_physical_plan(root, "q85").await;
    let lowering = lower_physical_plan_to_native(plain_plan);
    assert!(lowering.rewritten_count() > 0, "q85 should still lower against a plain DataFusion physical plan");

    let query_filter = vec!["q85".to_string()];
    let items = tpcds_parity_workload_items(
        root,
        sf1_tpcds_test_session_config(),
        sf1_native_optimizer_config(),
        Some(&query_filter),
    )
    .await
    .unwrap();
    let report = NativeTpcdsParityReport::collect(items, ParityFloatTolerance::exact());
    assert_eq!(report.rows.len(), 1);

    let row = &report.rows[0];
    assert_partial_native_parity_row(row, "q85", ParityComparisonMode::OrderedExact, 6, 1, 7);
}

#[tokio::test]
async fn tpcds_sf1_q93_matches_after_outer_join_filter_reassociation() {
    let Some(root) = sf1_fixture_root_or_skip() else {
        return;
    };
    let root = root.as_path();
    assert!(root.join("store_sales.parquet").is_file(), "TPC-DS SF1 fixture is missing at {}", root.display());

    let query_filter = vec!["q93".to_string()];
    let items = tpcds_parity_workload_items(
        root,
        sf1_tpcds_test_session_config(),
        sf1_native_optimizer_config(),
        Some(&query_filter),
    )
    .await
    .unwrap();
    let report = NativeTpcdsParityReport::collect(items, ParityFloatTolerance::exact());
    assert_eq!(report.rows.len(), 1);

    let row = &report.rows[0];
    assert_eq!(row.query_id, "q93");
    assert_eq!(row.cpu_planning_status, ParityStageStatus::Succeeded);
    assert_eq!(row.cpu_runtime_status, ParityStageStatus::Succeeded);
    assert_eq!(row.native_planning_outcome, NativeTpcdsParityPlanningOutcome::AcceptedReplaced);
    assert_eq!(row.native_reason_code, None);
    assert_eq!(row.native_unsupported_category, None);
    assert!(row.native_runtime_started, "q93 must execute through the native path");
    assert_eq!(row.native_runtime_outcome, NativeDataFusionRuntimeOutcome::Succeeded);
    assert_eq!(row.parity_status, ParityStatus::Matched);
    assert_eq!(row.comparison_mode, ParityComparisonMode::OrderedExact);
    assert_eq!(row.native_reservation_denied_count, Some(0));
    assert_eq!(row.native_retry_count, Some(0));
    assert_eq!(row.native_runtime_failure, None);
}

#[tokio::test]
async fn tpcds_sf1_case_residual_post_filter_queries_match_after_one_shot_admission() {
    let Some(root) = sf1_fixture_root_or_skip() else {
        return;
    };
    let root = root.as_path();
    assert!(root.join("customer.parquet").is_file(), "TPC-DS SF1 fixture is missing at {}", root.display());

    let query_filter = vec!["q04".to_string(), "q11".to_string(), "q31".to_string(), "q74".to_string()];
    let items = tpcds_parity_workload_items(
        root,
        sf1_tpcds_test_session_config(),
        sf1_native_optimizer_config(),
        Some(&query_filter),
    )
    .await
    .unwrap();
    let report = NativeTpcdsParityReport::collect(items, ParityFloatTolerance::exact());
    assert_eq!(report.rows.len(), 4);

    for (query_id, output_rows) in [("q04", 8), ("q11", 88), ("q31", 51), ("q74", 92)] {
        let row = report.rows.iter().find(|row| row.query_id == query_id).expect("query row should exist");
        let (fragments, boundaries) = match query_id {
            "q04" => (6, 24),
            "q11" => (4, 16),
            "q31" => (6, 20),
            "q74" => (4, 16),
            _ => unreachable!("covered query ids are exhaustive"),
        };
        assert_partial_native_parity_row(
            row,
            query_id,
            ParityComparisonMode::OrderedExact,
            output_rows,
            fragments,
            boundaries,
        );
    }
}

#[tokio::test]
async fn tpcds_sf1_q15_nullable_or_residual_matches_after_post_filter_admission() {
    let Some(root) = sf1_fixture_root_or_skip() else {
        return;
    };
    let root = root.as_path();
    assert!(root.join("catalog_sales.parquet").is_file(), "TPC-DS SF1 fixture is missing at {}", root.display());

    let query_filter = vec!["q15".to_string()];
    let items = tpcds_parity_workload_items(
        root,
        sf1_tpcds_test_session_config(),
        sf1_native_optimizer_config(),
        Some(&query_filter),
    )
    .await
    .unwrap();
    let report = NativeTpcdsParityReport::collect(items, ParityFloatTolerance::exact());
    assert_eq!(report.rows.len(), 1);

    let row = &report.rows[0];
    assert_eq!(row.query_id, "q15");
    assert_eq!(row.cpu_planning_status, ParityStageStatus::Succeeded);
    assert_eq!(row.cpu_runtime_status, ParityStageStatus::Succeeded);
    assert_eq!(row.native_planning_outcome, NativeTpcdsParityPlanningOutcome::AcceptedReplaced);
    assert_eq!(row.native_reason_code, None);
    assert_eq!(row.native_unsupported_category, None);
    assert!(row.native_runtime_started, "q15 must execute through the native path");
    assert_eq!(row.native_runtime_outcome, NativeDataFusionRuntimeOutcome::Succeeded);
    assert_eq!(row.parity_status, ParityStatus::Matched);
    assert_eq!(row.comparison_mode, ParityComparisonMode::OrderedExact);
    assert_eq!(row.cpu_output_rows, Some(100));
    assert_eq!(row.native_output_rows, Some(100));
    assert_eq!(row.native_reservation_denied_count, Some(0));
    assert_eq!(row.native_retry_count, Some(0));
    assert_eq!(row.native_runtime_failure, None);
}

#[tokio::test]
async fn tpcds_sf1_global_limit_join_aggregate_queries_match_after_limit_admission() {
    let Some(root) = sf1_fixture_root_or_skip() else {
        return;
    };
    let root = root.as_path();
    assert!(root.join("catalog_sales.parquet").is_file(), "TPC-DS SF1 fixture is missing at {}", root.display());

    let query_filter = vec![
        "q16".to_string(),
        "q32".to_string(),
        "q92".to_string(),
        "q94".to_string(),
        "q95".to_string(),
        "q96".to_string(),
        "q97".to_string(),
    ];
    let items = tpcds_parity_workload_items(
        root,
        sf1_tpcds_test_session_config(),
        sf1_native_optimizer_config(),
        Some(&query_filter),
    )
    .await
    .unwrap();
    let report = NativeTpcdsParityReport::collect(items, ParityFloatTolerance::exact());
    assert_eq!(report.rows.len(), 7);

    for (query_id, comparison_mode) in [
        ("q16", ParityComparisonMode::OrderedExact),
        ("q32", ParityComparisonMode::UnorderedExact),
        ("q92", ParityComparisonMode::OrderedExact),
        ("q94", ParityComparisonMode::OrderedExact),
        ("q95", ParityComparisonMode::OrderedExact),
        ("q96", ParityComparisonMode::OrderedExact),
        ("q97", ParityComparisonMode::UnorderedExact),
    ] {
        let row = report.rows.iter().find(|row| row.query_id == query_id).expect("query row should exist");
        assert_whole_native_parity_row(row, query_id, comparison_mode, 1);
    }
}

#[tokio::test]
async fn tpcds_sf1_q61_matches_after_physical_cross_join_admission() {
    let Some(root) = sf1_fixture_root_or_skip() else {
        return;
    };
    let root = root.as_path();
    assert!(root.join("store_sales.parquet").is_file(), "TPC-DS SF1 fixture is missing at {}", root.display());

    let lowering = lower_physical_plan_to_native(plain_tpcds_physical_plan(root, "q61").await);
    assert_eq!(lowering.rewritten_count(), 1, "q61 outcomes={:?}", lowering.report.outcomes());
    assert_eq!(lowering.rejected_count(), 0, "q61 outcomes={:?}", lowering.report.outcomes());
    let reader_pushdown_sources = lowering
        .source_diagnostics
        .iter()
        .filter(|diagnostics| diagnostics.predicate_disposition.as_str() == "reader_pushdown")
        .count();
    assert!(
        reader_pushdown_sources >= 2,
        "q61 decimal offset predicates should remain source reader pushdown; source_diagnostics={:?}",
        lowering.source_diagnostics
    );

    let query_filter = vec!["q61".to_string()];
    let items = tpcds_parity_workload_items(
        root,
        sf1_tpcds_test_session_config(),
        sf1_native_optimizer_config(),
        Some(&query_filter),
    )
    .await
    .unwrap();
    let report = NativeTpcdsParityReport::collect(items, ParityFloatTolerance::exact());
    assert_eq!(report.rows.len(), 1);

    let row = &report.rows[0];
    assert_whole_native_parity_row(row, "q61", ParityComparisonMode::OrderedExact, 1);
}

#[tokio::test]
async fn tpcds_sf1_q09_matches_after_scalar_nested_loop_left_join_cross_admission() {
    let Some(root) = sf1_fixture_root_or_skip() else {
        return;
    };
    let root = root.as_path();
    assert!(root.join("store_sales.parquet").is_file(), "TPC-DS SF1 fixture is missing at {}", root.display());

    let query_filter = vec!["q09".to_string()];
    let items = tpcds_parity_workload_items(
        root,
        sf1_tpcds_test_session_config(),
        sf1_native_optimizer_config(),
        Some(&query_filter),
    )
    .await
    .unwrap();
    let report = NativeTpcdsParityReport::collect(items, ParityFloatTolerance::exact());
    assert_eq!(report.rows.len(), 1);

    let row = &report.rows[0];
    assert_whole_native_parity_row(row, "q09", ParityComparisonMode::UnorderedExact, 1);
    assert_eq!(row.native_runtime_failure, None, "q09");
}

#[tokio::test]
async fn tpcds_sf1_root_full_sort_join_aggregate_queries_match_as_whole_native() {
    let Some(root) = sf1_fixture_root_or_skip() else {
        return;
    };
    let root = root.as_path();
    assert!(root.join("store_sales.parquet").is_file(), "TPC-DS SF1 fixture is missing at {}", root.display());

    let query_filter =
        vec!["q24".to_string(), "q31".to_string(), "q34".to_string(), "q64".to_string(), "q73".to_string()];
    let items = tpcds_parity_workload_items(
        root,
        sf1_tpcds_test_session_config(),
        sf1_native_optimizer_config(),
        Some(&query_filter),
    )
    .await
    .unwrap();
    let report = NativeTpcdsParityReport::collect(items, ParityFloatTolerance::exact());
    assert_eq!(report.rows.len(), 5);

    for (query_id, output_rows) in [("q24", 3), ("q31", 51), ("q34", 451), ("q64", 10), ("q73", 5)] {
        let row = report.rows.iter().find(|row| row.query_id == query_id).expect("query row should exist");
        if query_id == "q31" {
            assert_partial_native_parity_row(row, query_id, ParityComparisonMode::OrderedExact, output_rows, 6, 20);
        } else {
            assert_whole_native_parity_row(row, query_id, ParityComparisonMode::OrderedExact, output_rows);
        }
    }
}

#[test]
fn tpcds_sf1_current_partial_native_coverage_set_is_pinned() {
    run_tpcds_sf1_heavy_planning_test("tpcds_sf1_partial_native_coverage", || async {
        let Some(root) = sf1_fixture_root_or_skip() else {
            return;
        };
        let root = root.as_path();
        assert!(root.join("store_sales.parquet").is_file(), "TPC-DS SF1 fixture is missing at {}", root.display());

        let items = tpcds_datafusion_planning_workload_items_with_optimizer_config(
            root,
            tpcds_native_report_session_config(),
            sf1_native_optimizer_config(),
        )
        .await
        .unwrap();
        let report = NativeDataFusionPlanningReport::collect(items);
        let partial_native: BTreeSet<&str> = report
            .rows
            .iter()
            .filter(|row| row.post_native_execution_path == NativeDataFusionPostNativeExecutionPath::PartialNative)
            .map(|row| row.query_id.as_str())
            .collect();
        let cpu_only: BTreeSet<&str> = report
            .rows
            .iter()
            .filter(|row| row.post_native_execution_path == NativeDataFusionPostNativeExecutionPath::Cpu)
            .map(|row| row.query_id.as_str())
            .collect();
        let expected: BTreeSet<&str> =
            ["q04", "q06", "q08", "q11", "q31", "q41", "q47", "q54", "q57", "q59", "q74", "q83", "q85"]
                .into_iter()
                .collect();
        let cpu_expected: BTreeSet<&str> = BTreeSet::new();

        assert_eq!(partial_native, expected);
        assert_eq!(cpu_only, cpu_expected);
        for query_id in expected {
            let rows: Vec<_> = report.rows.iter().filter(|row| row.query_id == query_id).collect();
            assert!(!rows.is_empty(), "{query_id} should have planning rows");
            assert!(
                rows.iter().all(|row| row.native_plan_fragment_count > 0),
                "{query_id} should report native fragments"
            );
            assert!(
                rows.iter().all(|row| row.host_datafusion_boundary_count > 0),
                "{query_id} should report host DataFusion boundary operators"
            );
        }
        for query_id in [
            "q01", "q24", "q28", "q30", "q32", "q34", "q38", "q44", "q46", "q51", "q58", "q61", "q64", "q65", "q67",
            "q68", "q73", "q78", "q79", "q81", "q90", "q92", "q97", "q70", "q86", "q88", "q39",
        ] {
            let row = report
                .rows
                .iter()
                .find(|row| row.query_id == query_id)
                .unwrap_or_else(|| panic!("{query_id} is missing"));
            assert_eq!(row.outcome, NativeDataFusionPlanningOutcome::AcceptedReplaced, "{query_id}");
            assert_eq!(row.reason_code, None, "{query_id}");
            assert_eq!(row.unsupported_category, None, "{query_id}");
            assert_eq!(row.post_native_execution_path, NativeDataFusionPostNativeExecutionPath::Native, "{query_id}");
            assert_eq!(row.native_plan_fragment_count, 1, "{query_id}");
            assert_eq!(row.host_datafusion_boundary_count, 0, "{query_id}");
            assert!(!row.rejected_before_execution, "{query_id}");
        }
        let q39 = report.rows.iter().find(|row| row.query_id == "q39").unwrap_or_else(|| panic!("q39 is missing"));
        assert_eq!(q39.strict_bounded_compiled_graphs.len(), 1, "q39 should expose one accepted strict graph");
        let q39_graph = &q39.strict_bounded_compiled_graphs[0];
        assert_eq!(q39_graph.task_count, 47, "q39 compiled graph task count drifted");
        assert_eq!(q39_graph.edge_count, 51, "q39 compiled graph edge count drifted");
        assert_eq!(q39_graph.barrier_count, 23, "q39 compiled graph barrier count drifted");
        assert_eq!(q39_graph.resource_count, 32, "q39 compiled graph resource count drifted");
        assert_eq!(q39_graph.gate_count, 50, "q39 compiled graph gate count drifted");
        assert_eq!(
            q39_graph.sink_kind, "host_arrow_materialization",
            "q39 compiled graph should sink through host Arrow materialization"
        );
        assert_eq!(q39_graph.sink_gate_count, 1, "q39 compiled graph sink gate count drifted");
        for required_contract in [
            "host_welford_input_materialization_preflight",
            "host_welford_output_upload_reservation",
            "final_full_sort_output_cardinality_bound",
            "final_sort_contract",
            "query_engine_explicit_sink_materialization_reservation",
        ] {
            assert!(
                q39.strict_bounded_contract_rows.iter().any(|gate| {
                    gate.required_contract == required_contract && gate.contract_status.as_str() == "modeled"
                }),
                "q39 should expose modeled exact statistical/final-sort gate {required_contract}"
            );
        }
        for query_id in ["q34", "q73"] {
            let row = report
                .rows
                .iter()
                .find(|row| row.query_id == query_id)
                .unwrap_or_else(|| panic!("{query_id} is missing"));
            for (reservation_kind, required_contract) in [
                (
                    "join_output",
                    "dataframe_gather_output_reservation_size_bytes+projected_cross_join_reservation_size_bytes",
                ),
                ("operator_output", "aggregate_contract"),
                ("final_sort", "final_sort_contract"),
                ("materialization", "query_engine_explicit_sink_materialization_reservation"),
            ] {
                assert!(
                    row.strict_bounded_contract_rows.iter().any(|gate| {
                        gate.reservation_kind.as_str() == reservation_kind
                            && gate.required_contract == required_contract
                            && gate.contract_status.as_str() == "modeled"
                    }),
                    "{query_id} should expose modeled root full-sort strict gate {required_contract}"
                );
            }
        }
        for (query_id, expected_contract) in [
            ("q24", "root_full_sort_keyless_or_cross_join_subcontract"),
            ("q64", "root_full_sort_repeated_projection_fanout_subcontract"),
        ] {
            let row = report
                .rows
                .iter()
                .find(|row| row.query_id == query_id)
                .unwrap_or_else(|| panic!("{query_id} is missing"));
            assert!(
                row.strict_bounded_contract_rows.iter().any(|gate| {
                    gate.reservation_kind.as_str() == "capability_contract"
                        && gate.memory_boundary == expected_contract
                        && gate.required_contract == expected_contract
                        && gate.contract_status.as_str() == "modeled"
                }),
                "{query_id} should expose root full-sort subcontract {expected_contract} in strict report rows"
            );
            for reservation_kind in ["join_output", "operator_output", "final_sort", "materialization"] {
                assert!(
                    row.strict_bounded_contract_rows.iter().any(|gate| {
                        gate.reservation_kind.as_str() == reservation_kind && gate.contract_status.as_str() == "modeled"
                    }),
                    "{query_id} should expose modeled root full-sort strict gate kind {reservation_kind}"
                );
            }
        }
        let q67 = report.rows.iter().find(|row| row.query_id == "q67").unwrap_or_else(|| panic!("q67 is missing"));
        assert!(
            q67.strict_bounded_contract_rows.iter().any(|gate| {
                gate.reservation_kind.as_str() == "capability_contract"
                    && gate.memory_boundary == "rollup_rank_topk_wide_final_rollup_subcontract"
                    && gate.required_contract == "rollup_rank_topk_wide_final_rollup_subcontract"
                    && gate.contract_status.as_str() == "modeled"
            }),
            "q67 should expose the wide rollup/rank/top-k subcontract in strict report rows"
        );
        for (query_id, expected_contract) in [
            ("q01", "root_topk_sort_fetch_hash_join_input_subcontract"),
            ("q58", "root_topk_keyed_filtered_semi_join_subcontract"),
            ("q78", "root_topk_keyed_filtered_semi_join_subcontract"),
            ("q61", "root_topk_keyless_or_cross_join_subcontract"),
            ("q90", "root_topk_keyless_or_cross_join_subcontract"),
            ("q51", "root_topk_window_subcontract"),
            ("q44", "root_topk_window_subcontract"),
            ("q44", "root_topk_keyless_outer_cross_window_subcontract"),
        ] {
            let row = report
                .rows
                .iter()
                .find(|row| row.query_id == query_id)
                .unwrap_or_else(|| panic!("{query_id} is missing"));
            assert!(
                row.strict_bounded_contract_rows.iter().any(|gate| {
                    gate.reservation_kind.as_str() == "capability_contract"
                        && gate.memory_boundary == expected_contract
                        && gate.required_contract == expected_contract
                        && gate.contract_status.as_str() == "modeled"
                }),
                "{query_id} should expose root TopK subcontract {expected_contract} in strict report rows"
            );
            let [compiled_graph] = row.strict_bounded_compiled_graphs.as_slice() else {
                panic!("{query_id} should report one strict bounded compiled graph");
            };
            assert_eq!(compiled_graph.sink_kind, "host_arrow_materialization", "{query_id}");
            assert_eq!(compiled_graph.sink_gate_count, 1, "{query_id}");
        }
        for query_id in ["q44", "q51", "q67"] {
            let row = report
                .rows
                .iter()
                .find(|row| row.query_id == query_id)
                .unwrap_or_else(|| panic!("{query_id} is missing"));
            for expected_contract in ["window_contract", "query_engine_final_topk_output_boundary"] {
                assert!(
                    row.strict_bounded_contract_rows.iter().any(|gate| {
                        gate.reservation_kind.as_str() == "operator_output"
                            && gate.required_contract == expected_contract
                            && gate.contract_status.as_str() == "modeled"
                    }),
                    "{query_id} should expose window operator contract {expected_contract} in strict report rows"
                );
            }
        }
        let q67 = report.rows.iter().find(|row| row.query_id == "q67").unwrap_or_else(|| panic!("q67 is missing"));
        assert!(
            q67.strict_bounded_contract_rows.iter().any(|gate| {
                gate.reservation_kind.as_str() == "operator_output"
                    && gate.required_contract == "query_engine_rank_filter_output_boundary"
                    && gate.contract_status.as_str() == "modeled"
            }),
            "q67 should expose rank/filter output boundary in strict report rows"
        );
        let q51 = report.rows.iter().find(|row| row.query_id == "q51").unwrap_or_else(|| panic!("q51 is missing"));
        assert!(
            q51.strict_bounded_contract_rows.iter().any(|gate| {
                gate.reservation_kind.as_str() == "capability_contract"
                    && gate.memory_boundary == "root_topk_window_full_outer_join_subcontract"
                    && gate.required_contract == "root_topk_window_full_outer_join_subcontract"
                    && gate.contract_status.as_str() == "modeled"
            }),
            "q51 should expose full-outer window subcontract in strict report rows"
        );
        let q44 = report.rows.iter().find(|row| row.query_id == "q44").unwrap_or_else(|| panic!("q44 is missing"));
        for (reservation_kind, required_contract) in [
            ("gather_map", "join_gather_map_reservation_size_bytes"),
            (
                "join_output",
                "dataframe_gather_output_reservation_size_bytes+projected_cross_join_reservation_size_bytes",
            ),
        ] {
            assert!(
                q44.strict_bounded_contract_rows.iter().any(|gate| {
                    gate.reservation_kind.as_str() == reservation_kind
                        && gate.required_contract == required_contract
                        && gate.contract_status.as_str() == "modeled"
                }),
                "q44 should expose keyless outer-cross contract {required_contract} in strict report rows"
            );
        }
        for query_id in ["q30", "q65", "q81"] {
            let row = report
                .rows
                .iter()
                .find(|row| row.query_id == query_id)
                .unwrap_or_else(|| panic!("{query_id} is missing"));
            assert!(
                row.native_optimizer.shared_subplan_materialization.considered_count > 0,
                "{query_id} should retain shared-subplan discovery diagnostics"
            );
            assert!(
                row.native_optimizer.dynamic_filter_route_discovery.route_count > 0,
                "{query_id} should retain source-reader reduction proof diagnostics"
            );
            let [compiled_graph] = row.strict_bounded_compiled_graphs.as_slice() else {
                panic!("{query_id} should report one strict bounded compiled graph");
            };
            assert_eq!(compiled_graph.sink_kind, "host_arrow_materialization", "{query_id}");
            assert_eq!(compiled_graph.sink_gate_count, 1, "{query_id}");
        }
        for query_id in ["q30", "q81"] {
            let row = report
                .rows
                .iter()
                .find(|row| row.query_id == query_id)
                .unwrap_or_else(|| panic!("{query_id} is missing"));
            assert!(
                row.native_optimizer.local_parquet_source_reuse.applied_count > 0,
                "{query_id} should retain local source reuse diagnostics"
            );
        }
    });
}

#[tokio::test]
async fn tpcds_sf1_q44_keyless_outer_cross_window_matches_as_whole_native() {
    let Some(root) = sf1_fixture_root_or_skip() else {
        return;
    };
    let root = root.as_path();
    assert!(root.join("store_sales.parquet").is_file(), "TPC-DS SF1 fixture is missing at {}", root.display());

    let query_filter = vec!["q44".to_string()];
    let items = tpcds_parity_workload_items(
        root,
        sf1_tpcds_test_session_config(),
        sf1_native_optimizer_config(),
        Some(&query_filter),
    )
    .await
    .unwrap();
    let report = NativeTpcdsParityReport::collect(items, ParityFloatTolerance::exact());
    assert_eq!(report.rows.len(), 1);

    let row = &report.rows[0];
    assert_whole_native_parity_row(row, "q44", ParityComparisonMode::OrderedExact, 10);
}

#[tokio::test]
async fn tpcds_sf1_bitwise_grouping_window_queries_match_as_whole_native() {
    let Some(root) = sf1_fixture_root_or_skip() else {
        return;
    };
    let root = root.as_path();
    assert!(root.join("web_sales.parquet").is_file(), "TPC-DS SF1 fixture is missing at {}", root.display());

    let query_filter = vec!["q70".to_string(), "q86".to_string()];
    let items = tpcds_parity_workload_items(
        root,
        sf1_tpcds_test_session_config(),
        sf1_native_optimizer_config(),
        Some(&query_filter),
    )
    .await
    .unwrap();
    let report = NativeTpcdsParityReport::collect(items, ParityFloatTolerance::exact());
    assert_eq!(report.rows.len(), 2);

    let q70 = report.rows.iter().find(|row| row.query_id == "q70").expect("q70 row should exist");
    assert_whole_native_parity_row(q70, "q70", ParityComparisonMode::OrderedExact, 3);

    let q86 = report.rows.iter().find(|row| row.query_id == "q86").expect("q86 row should exist");
    assert_whole_native_parity_row(q86, "q86", ParityComparisonMode::OrderedExact, 100);
}

#[tokio::test]
async fn tpcds_sf1_abs_window_filter_queries_match_as_whole_native() {
    let Some(root) = sf1_fixture_root_or_skip() else {
        return;
    };
    let root = root.as_path();
    assert!(root.join("store_sales.parquet").is_file(), "TPC-DS SF1 fixture is missing at {}", root.display());

    let query_filter = vec!["q53".to_string(), "q63".to_string(), "q89".to_string()];
    let items = tpcds_parity_workload_items(
        root,
        sf1_tpcds_test_session_config(),
        sf1_native_optimizer_config(),
        Some(&query_filter),
    )
    .await
    .unwrap();
    let report = NativeTpcdsParityReport::collect(items, ParityFloatTolerance::exact());
    assert_eq!(report.rows.len(), 3);

    for (query_id, output_rows) in [("q53", 100), ("q63", 100), ("q89", 100)] {
        let row = report.rows.iter().find(|row| row.query_id == query_id).expect("query row should exist");
        assert_eq!(row.cpu_planning_status, ParityStageStatus::Succeeded, "{query_id}");
        assert_eq!(row.cpu_runtime_status, ParityStageStatus::Succeeded, "{query_id}");
        assert_eq!(row.native_planning_outcome, NativeTpcdsParityPlanningOutcome::AcceptedReplaced, "{query_id}");
        assert_eq!(row.post_native_execution_path, NativeDataFusionPostNativeExecutionPath::Native, "{query_id}");
        assert_eq!(row.native_plan_fragment_count, 1, "{query_id}");
        assert_eq!(row.host_datafusion_boundary_count, 0, "{query_id}");
        assert!(row.native_runtime_started, "{query_id} must execute through the native path");
        assert_eq!(row.native_runtime_outcome, NativeDataFusionRuntimeOutcome::Succeeded, "{query_id}");
        assert_eq!(row.parity_status, ParityStatus::Matched, "{query_id}");
        assert_eq!(row.comparison_mode, ParityComparisonMode::OrderedExact, "{query_id}");
        assert_eq!(row.cpu_output_rows, Some(output_rows), "{query_id}");
        assert_eq!(row.native_output_rows, Some(output_rows), "{query_id}");
        assert_eq!(row.native_reservation_denied_count, Some(0), "{query_id}");
        assert_eq!(row.native_retry_count, Some(0), "{query_id}");
        assert_eq!(row.native_runtime_failure, None, "{query_id}");
    }
}

#[test]
fn tpcds_sf1_residual_root_topk_rejections_are_owned_by_root_topk_analyzer() {
    run_tpcds_sf1_heavy_planning_test("tpcds_sf1_root_topk_rejection_owner", || async {
        let Some(root) = sf1_fixture_root_or_skip() else {
            return;
        };
        let root = root.as_path();
        assert!(root.join("store_sales.parquet").is_file(), "TPC-DS SF1 fixture is missing at {}", root.display());

        let items = tpcds_datafusion_planning_workload_items_with_optimizer_config(
            root,
            tpcds_native_report_session_config(),
            sf1_native_optimizer_config(),
        )
        .await
        .unwrap();
        let report = NativeDataFusionPlanningReport::collect(items);

        let residuals = report
            .rows
            .iter()
            .filter(|row| {
                row.strict_bounded_contract_rows.iter().any(|gate| {
                    gate.owner.to_string() == "admission:root_topk_join_aggregate_admission"
                        && gate.contract_status.as_str() == "rejected_by_capability"
                })
            })
            .map(|row| row.query_id.as_str())
            .collect::<Vec<_>>();
        assert!(residuals.is_empty(), "root TopK admission should not own residual SF1 rejections: {residuals:?}");
    });
}

#[tokio::test]
async fn tpcds_sf1_rank_window_queries_match_current_admission_contract() {
    let Some(root) = sf1_fixture_root_or_skip() else {
        return;
    };
    let root = root.as_path();
    assert!(root.join("store_sales.parquet").is_file(), "TPC-DS SF1 fixture is missing at {}", root.display());

    let query_filter = vec!["q36".to_string(), "q47".to_string(), "q57".to_string(), "q67".to_string()];
    let items = tpcds_parity_workload_items(
        root,
        sf1_tpcds_test_session_config(),
        sf1_native_optimizer_config(),
        Some(&query_filter),
    )
    .await
    .unwrap();
    let report = NativeTpcdsParityReport::collect(items, ParityFloatTolerance::exact());
    assert_eq!(report.rows.len(), 4);

    let q36 = report.rows.iter().find(|row| row.query_id == "q36").expect("q36 row should exist");
    assert_whole_native_parity_row(q36, "q36", ParityComparisonMode::OrderedExact, 100);

    let q67_lowering = lower_physical_plan_to_native(plain_tpcds_physical_plan(root, "q67").await);
    assert_eq!(q67_lowering.rewritten_count(), 1, "q67 outcomes={:?}", q67_lowering.report.outcomes());
    assert_eq!(q67_lowering.rejected_count(), 0, "q67 outcomes={:?}", q67_lowering.report.outcomes());
    let q67_optimized = nexus_query_engine::optimizer::optimize_query_plan(
        q67_lowering.plans[0].clone(),
        &nexus_query_engine::optimizer::NativeOptimizerOptions::default(),
    )
    .plan;
    let q67_analysis = q67_optimized.capability_analysis();
    assert!(
        !q67_analysis.admits_root_topk_join_aggregate_pipeline(),
        "q67 must not be promoted through the generic root TopK envelope"
    );
    assert!(
        q67_analysis.admits_rollup_rank_topk_after_aggregate_join_pushdown_pipeline(),
        "q67 must be promoted through the explicit rollup/rank/top-k envelope"
    );
    let nexus_query_engine::capability::CapabilityEnvelopeAdmission::Admitted { envelope } =
        q67_analysis.strict_bounded_envelope_admission()
    else {
        panic!("q67 should select a strict bounded envelope: {:?}", q67_analysis.first_rejection_projection());
    };
    assert_eq!(envelope.as_str(), "rollup_rank_topk_after_aggregate_join_pushdown");

    for query_id in ["q47", "q57"] {
        let row = report.rows.iter().find(|row| row.query_id == query_id).expect("query row should exist");
        assert_partial_native_parity_row(row, query_id, ParityComparisonMode::OrderedExact, 100, 3, 21);
    }

    let row = report.rows.iter().find(|row| row.query_id == "q67").expect("query row should exist");
    assert_whole_native_parity_row(row, "q67", ParityComparisonMode::OrderedExact, 100);
}

#[tokio::test]
async fn tpcds_sf1_partition_aggregate_window_queries_match_as_whole_native() {
    let Some(root) = sf1_fixture_root_or_skip() else {
        return;
    };
    let root = root.as_path();
    assert!(root.join("store_sales.parquet").is_file(), "TPC-DS SF1 fixture is missing at {}", root.display());

    let query_filter = vec!["q12".to_string(), "q20".to_string(), "q98".to_string()];
    let items = tpcds_parity_workload_items(
        root,
        sf1_tpcds_test_session_config(),
        sf1_native_optimizer_config(),
        Some(&query_filter),
    )
    .await
    .unwrap();
    let report = NativeTpcdsParityReport::collect(items, ParityFloatTolerance::exact());
    assert_eq!(report.rows.len(), 3);

    for (query_id, output_rows) in [("q12", 100), ("q20", 100), ("q98", 2516)] {
        let row = report.rows.iter().find(|row| row.query_id == query_id).expect("query row should exist");
        assert_eq!(row.cpu_planning_status, ParityStageStatus::Succeeded, "{query_id}");
        assert_eq!(row.cpu_runtime_status, ParityStageStatus::Succeeded, "{query_id}");
        assert_eq!(row.native_planning_outcome, NativeTpcdsParityPlanningOutcome::AcceptedReplaced, "{query_id}");
        assert_eq!(row.post_native_execution_path, NativeDataFusionPostNativeExecutionPath::Native, "{query_id}");
        assert_eq!(row.native_plan_fragment_count, 1, "{query_id}");
        assert_eq!(row.host_datafusion_boundary_count, 0, "{query_id}");
        assert!(row.native_runtime_started, "{query_id} must execute through the native path");
        assert_eq!(row.native_runtime_outcome, NativeDataFusionRuntimeOutcome::Succeeded, "{query_id}");
        assert_eq!(row.parity_status, ParityStatus::Matched, "{query_id}");
        assert_eq!(row.comparison_mode, ParityComparisonMode::OrderedExact, "{query_id}");
        assert_eq!(row.cpu_output_rows, Some(output_rows), "{query_id}");
        assert_eq!(row.native_output_rows, Some(output_rows), "{query_id}");
        assert_eq!(row.native_reservation_denied_count, Some(0), "{query_id}");
        assert_eq!(row.native_retry_count, Some(0), "{query_id}");
        assert_eq!(row.native_runtime_failure, None, "{query_id}");
    }
}

#[tokio::test]
async fn tpcds_sf1_cumulative_aggregate_window_query_matches_as_whole_native() {
    let Some(root) = sf1_fixture_root_or_skip() else {
        return;
    };
    let root = root.as_path();
    assert!(root.join("store_sales.parquet").is_file(), "TPC-DS SF1 fixture is missing at {}", root.display());

    let query_filter = vec!["q51".to_string()];
    let items = tpcds_parity_workload_items(
        root,
        sf1_tpcds_test_session_config(),
        sf1_native_optimizer_config(),
        Some(&query_filter),
    )
    .await
    .unwrap();
    let report = NativeTpcdsParityReport::collect(items, ParityFloatTolerance::exact());
    assert_eq!(report.rows.len(), 1);

    let (query_id, output_rows) = ("q51", 100);
    let row = report.rows.iter().find(|row| row.query_id == query_id).expect("query row should exist");
    assert_whole_native_parity_row(row, query_id, ParityComparisonMode::OrderedExact, output_rows);
}

#[tokio::test]
async fn tpcds_sf1_rank_window_union_query_matches_as_whole_native() {
    let Some(root) = sf1_fixture_root_or_skip() else {
        return;
    };
    let root = root.as_path();
    assert!(root.join("store_sales.parquet").is_file(), "TPC-DS SF1 fixture is missing at {}", root.display());

    let query_filter = vec!["q49".to_string()];
    let items = tpcds_parity_workload_items(
        root,
        sf1_tpcds_test_session_config(),
        sf1_native_optimizer_config(),
        Some(&query_filter),
    )
    .await
    .unwrap();
    let report = NativeTpcdsParityReport::collect(items, ParityFloatTolerance::exact());
    assert_eq!(report.rows.len(), 1);

    let row = report.rows.iter().find(|row| row.query_id == "q49").expect("q49 row should exist");
    assert_eq!(row.cpu_planning_status, ParityStageStatus::Succeeded);
    assert_eq!(row.cpu_runtime_status, ParityStageStatus::Succeeded);
    assert_eq!(row.native_planning_outcome, NativeTpcdsParityPlanningOutcome::AcceptedReplaced);
    assert_eq!(row.native_reason_code, None);
    assert_eq!(row.native_unsupported_category, None);
    assert_eq!(row.post_native_execution_path, NativeDataFusionPostNativeExecutionPath::Native);
    assert_eq!(row.native_plan_fragment_count, 1);
    assert_eq!(row.host_datafusion_boundary_count, 0);
    assert!(row.native_runtime_started, "q49 must execute through the native path");
    assert_eq!(row.native_runtime_outcome, NativeDataFusionRuntimeOutcome::Succeeded);
    assert_eq!(row.parity_status, ParityStatus::Matched);
    assert_eq!(row.comparison_mode, ParityComparisonMode::OrderedExact);
    assert_eq!(row.cpu_output_rows, Some(32));
    assert_eq!(row.native_output_rows, Some(32));
    assert_eq!(row.native_reservation_denied_count, Some(0));
    assert_eq!(row.native_retry_count, Some(0));
    assert_eq!(row.native_runtime_failure, None);
}

#[tokio::test]
async fn tpcds_sf1_round_decimal_segment_query_matches_with_string_key_hash_prebuild_fragments_on_cpu() {
    let Some(root) = sf1_fixture_root_or_skip() else {
        return;
    };
    let root = root.as_path();
    assert!(root.join("store_sales.parquet").is_file(), "TPC-DS SF1 fixture is missing at {}", root.display());

    let query_filter = vec!["q54".to_string()];
    let items = tpcds_parity_workload_items(
        root,
        sf1_tpcds_test_session_config(),
        sf1_native_optimizer_config(),
        Some(&query_filter),
    )
    .await
    .unwrap();
    let report = NativeTpcdsParityReport::collect(items, ParityFloatTolerance::exact());
    assert_eq!(report.rows.len(), 1);

    let row = report.rows.iter().find(|row| row.query_id == "q54").expect("q54 row should exist");
    assert_partial_native_parity_row(row, "q54", ParityComparisonMode::OrderedExact, 1, 3, 12);
}

#[tokio::test]
async fn tpcds_sf1_mark_join_queries_match_as_whole_native() {
    let Some(root) = sf1_fixture_root_or_skip() else {
        return;
    };
    let root = root.as_path();
    assert!(root.join("store_sales.parquet").is_file(), "TPC-DS SF1 fixture is missing at {}", root.display());

    let query_filter = vec!["q10".to_string(), "q35".to_string(), "q45".to_string()];
    let items = tpcds_parity_workload_items(
        root,
        sf1_tpcds_test_session_config(),
        sf1_native_optimizer_config(),
        Some(&query_filter),
    )
    .await
    .unwrap();
    let report = NativeTpcdsParityReport::collect(items, ParityFloatTolerance::exact());
    assert_eq!(report.rows.len(), 3);

    for (query_id, output_rows) in [("q10", 5), ("q35", 100), ("q45", 19)] {
        let row = report.rows.iter().find(|row| row.query_id == query_id).expect("query row should exist");
        assert_eq!(row.cpu_planning_status, ParityStageStatus::Succeeded, "{query_id}");
        assert_eq!(row.cpu_runtime_status, ParityStageStatus::Succeeded, "{query_id}");
        assert_eq!(row.native_planning_outcome, NativeTpcdsParityPlanningOutcome::AcceptedReplaced, "{query_id}");
        assert_eq!(row.post_native_execution_path, NativeDataFusionPostNativeExecutionPath::Native, "{query_id}");
        assert_eq!(row.native_plan_fragment_count, 1, "{query_id}");
        assert_eq!(row.host_datafusion_boundary_count, 0, "{query_id}");
        assert!(row.native_runtime_started, "{query_id} must execute through the native path");
        assert_eq!(row.native_runtime_outcome, NativeDataFusionRuntimeOutcome::Succeeded, "{query_id}");
        assert_eq!(row.parity_status, ParityStatus::Matched, "{query_id}");
        assert_eq!(row.comparison_mode, ParityComparisonMode::OrderedExact, "{query_id}");
        assert_eq!(row.cpu_output_rows, Some(output_rows), "{query_id}");
        assert_eq!(row.native_output_rows, Some(output_rows), "{query_id}");
        assert_eq!(row.native_reservation_denied_count, Some(0), "{query_id}");
        assert_eq!(row.native_retry_count, Some(0), "{query_id}");
        assert_eq!(row.native_runtime_failure, None, "{query_id}");
    }
}

#[tokio::test]
async fn tpcds_sf1_host_arrow_join_boundary_queries_match_with_partial_native_children() {
    let Some(root) = sf1_fixture_root_or_skip() else {
        return;
    };
    let root = root.as_path();
    assert!(root.join("store_sales.parquet").is_file(), "TPC-DS SF1 fixture is missing at {}", root.display());

    let query_filter = vec!["q17".to_string()];
    let items = tpcds_parity_workload_items(
        root,
        sf1_tpcds_test_session_config(),
        sf1_native_optimizer_config(),
        Some(&query_filter),
    )
    .await
    .unwrap();
    let report = NativeTpcdsParityReport::collect(items, ParityFloatTolerance::exact());
    assert_eq!(report.rows.len(), 1);

    let (query_id, output_rows) = ("q17", 1);
    let row = report.rows.iter().find(|row| row.query_id == query_id).expect("query row should exist");
    assert_eq!(row.cpu_planning_status, ParityStageStatus::Succeeded, "{query_id}");
    assert_eq!(row.cpu_runtime_status, ParityStageStatus::Succeeded, "{query_id}");
    assert_eq!(row.native_planning_outcome, NativeTpcdsParityPlanningOutcome::AcceptedReplaced, "{query_id}");
    assert_eq!(row.native_reason_code, None, "{query_id}");
    assert_eq!(row.native_unsupported_category, None, "{query_id}");
    assert!(row.native_runtime_started, "{query_id} must execute with partial native children");
    assert_eq!(row.native_runtime_outcome, NativeDataFusionRuntimeOutcome::Succeeded, "{query_id}");
    assert_eq!(row.parity_status, ParityStatus::Matched, "{query_id}");
    assert_eq!(row.comparison_mode, ParityComparisonMode::OrderedExact, "{query_id}");
    assert_eq!(row.cpu_output_rows, Some(output_rows), "{query_id}");
    assert_eq!(row.native_output_rows, Some(output_rows), "{query_id}");
    assert_eq!(row.native_reservation_denied_count, Some(0), "{query_id}");
    assert_eq!(row.native_retry_count, Some(0), "{query_id}");
    assert_eq!(row.native_runtime_failure, None, "{query_id}");
}

#[tokio::test]
async fn tpcds_sf1_distributed_aggregate_queries_match_as_whole_native() {
    let Some(root) = sf1_fixture_root_or_skip() else {
        return;
    };
    let root = root.as_path();
    assert!(root.join("store_sales.parquet").is_file(), "TPC-DS SF1 fixture is missing at {}", root.display());

    let query_filter = vec![
        "q05".to_string(),
        "q14".to_string(),
        "q18".to_string(),
        "q22".to_string(),
        "q77".to_string(),
        "q80".to_string(),
    ];
    let items = tpcds_parity_workload_items(
        root,
        sf1_tpcds_test_session_config(),
        sf1_native_optimizer_config(),
        Some(&query_filter),
    )
    .await
    .unwrap();
    let report = NativeTpcdsParityReport::collect(items, ParityFloatTolerance::exact());
    assert_eq!(report.rows.len(), 6);

    for (query_id, output_rows) in [("q05", 100), ("q14", 100), ("q18", 100), ("q22", 100), ("q77", 44), ("q80", 100)] {
        let row = report.rows.iter().find(|row| row.query_id == query_id).expect("query row should exist");
        assert_eq!(row.cpu_planning_status, ParityStageStatus::Succeeded, "{query_id}");
        assert_eq!(row.cpu_runtime_status, ParityStageStatus::Succeeded, "{query_id}");
        assert_eq!(row.native_planning_outcome, NativeTpcdsParityPlanningOutcome::AcceptedReplaced, "{query_id}");
        assert_eq!(row.native_reason_code, None, "{query_id}");
        assert_eq!(row.native_unsupported_category, None, "{query_id}");
        assert_eq!(row.post_native_execution_path, NativeDataFusionPostNativeExecutionPath::Native, "{query_id}");
        assert_eq!(row.native_plan_fragment_count, 1, "{query_id}");
        assert_eq!(row.host_datafusion_boundary_count, 0, "{query_id}");
        assert!(row.native_runtime_started, "{query_id} must execute as one whole native subtree");
        assert_eq!(row.native_runtime_outcome, NativeDataFusionRuntimeOutcome::Succeeded, "{query_id}");
        assert_eq!(row.parity_status, ParityStatus::Matched, "{query_id}");
        assert_eq!(row.comparison_mode, ParityComparisonMode::OrderedExact, "{query_id}");
        assert_eq!(row.cpu_output_rows, Some(output_rows), "{query_id}");
        assert_eq!(row.native_output_rows, Some(output_rows), "{query_id}");
        assert_eq!(row.native_reservation_denied_count, Some(0), "{query_id}");
        assert_eq!(row.native_retry_count, Some(0), "{query_id}");
        assert_eq!(row.native_runtime_failure, None, "{query_id}");
    }
}

#[tokio::test]
async fn tpcds_sf1_distributed_aggregate_lowering_admits_shared_resident_grouping_sets() {
    let Some(root) = sf1_fixture_root_or_skip() else {
        return;
    };
    let root = root.as_path();
    assert!(root.join("store_sales.parquet").is_file(), "TPC-DS SF1 fixture is missing at {}", root.display());

    for query_id in ["q05", "q18"] {
        let lowering = lower_physical_plan_to_native(plain_tpcds_physical_plan(root, query_id).await);
        assert_eq!(lowering.rewritten_count(), 1, "{query_id} outcomes={:?}", lowering.report.outcomes());
        assert_eq!(lowering.rejected_count(), 0, "{query_id} outcomes={:?}", lowering.report.outcomes());

        let optimized = nexus_query_engine::optimizer::optimize_query_plan(
            lowering.plans[0].clone(),
            &nexus_query_engine::optimizer::NativeOptimizerOptions::default(),
        )
        .plan;
        let rendered = optimized.to_string();
        assert!(rendered.contains("Aggregate"), "{query_id} must retain native aggregate lowering:\n{rendered}");
        let admission = nexus_query_engine::admission::strict_bounded_pipeline_admission(&optimized)
            .expect("strict admission should not fail");
        assert!(admission.is_some(), "{query_id} strict bounded admission should accept the whole native plan");

        if query_id == "q05" {
            assert!(rendered.contains("CONCAT("), "q05 should lower scalar concat into native projection:\n{rendered}");
        }
        if query_id == "q18" {
            assert!(
                rendered.contains("__cudf_preagg_avg_"),
                "q18 grouping-set AVG should lower to explicit native SUM/COUNT state:\n{rendered}"
            );
            assert!(
                rendered.contains("__grouping_id"),
                "q18 grouping-set AVG final projection must preserve the grouping id output:\n{rendered}"
            );
            assert!(
                rendered.contains("__cudf_preagg_arg_"),
                "q18 AVG inputs should be materialized as reusable native aggregate arguments:\n{rendered}"
            );
        }
    }
}

/// Assert that a single TPC-DS SF1 UNION query lowers to one whole native
/// subtree and matches plain DataFusion exactly.
///
/// This is the per-query body behind the `tpcds_sf1_union_q*_matches_as_whole_native`
/// tests below. It was factored out of a former single test that looped over all
/// nine UNION queries in one `#[tokio::test]` (~53s, the GPU lane's single longest
/// test). One test per query lets nextest schedule them across the `gpu` group's
/// threads and makes a failure name the offending query instead of the whole set.
async fn assert_tpcds_sf1_union_query_matches_as_whole_native(query_id: &str, expected_output_rows: usize) {
    let Some(root) = sf1_fixture_root_or_skip() else {
        return;
    };
    let root = root.as_path();
    assert!(root.join("store_sales.parquet").is_file(), "TPC-DS SF1 fixture is missing at {}", root.display());

    let items = tpcds_parity_workload_items(
        root,
        sf1_tpcds_test_session_config(),
        sf1_native_optimizer_config(),
        Some(&[query_id.to_string()]),
    )
    .await
    .unwrap();
    let report = NativeTpcdsParityReport::collect(items, ParityFloatTolerance::exact());
    assert_eq!(report.rows.len(), 1, "{query_id}");

    let row = report.rows.iter().find(|row| row.query_id == query_id).expect("query row should exist");
    assert_eq!(row.cpu_planning_status, ParityStageStatus::Succeeded, "{query_id}");
    assert_eq!(row.cpu_runtime_status, ParityStageStatus::Succeeded, "{query_id}");
    assert_eq!(row.native_planning_outcome, NativeTpcdsParityPlanningOutcome::AcceptedReplaced, "{query_id}");
    assert_eq!(row.native_reason_code, None, "{query_id}");
    assert_eq!(row.native_unsupported_category, None, "{query_id}");
    assert_eq!(row.post_native_execution_path, NativeDataFusionPostNativeExecutionPath::Native, "{query_id}");
    assert_eq!(row.native_plan_fragment_count, 1, "{query_id}");
    assert_eq!(row.host_datafusion_boundary_count, 0, "{query_id}");
    assert!(row.native_runtime_started, "{query_id} must execute as one whole native subtree");
    assert_eq!(row.native_runtime_outcome, NativeDataFusionRuntimeOutcome::Succeeded, "{query_id}");
    assert_eq!(row.parity_status, ParityStatus::Matched, "{query_id}");
    assert_eq!(row.comparison_mode, ParityComparisonMode::OrderedExact, "{query_id}");
    assert_eq!(row.cpu_output_rows, Some(expected_output_rows), "{query_id}");
    assert_eq!(row.native_output_rows, Some(expected_output_rows), "{query_id}");
    assert_eq!(row.native_reservation_denied_count, Some(0), "{query_id}");
    assert_eq!(row.native_retry_count, Some(0), "{query_id}");
    assert_eq!(row.native_runtime_failure, None, "{query_id}");
}

macro_rules! tpcds_sf1_union_query_parity_test {
    ($name:ident, $query_id:literal, $expected_output_rows:expr) => {
        #[tokio::test]
        async fn $name() {
            assert_tpcds_sf1_union_query_matches_as_whole_native($query_id, $expected_output_rows).await;
        }
    };
}

tpcds_sf1_union_query_parity_test!(tpcds_sf1_union_q23_matches_as_whole_native, "q23", 4);
tpcds_sf1_union_query_parity_test!(tpcds_sf1_union_q27_matches_as_whole_native, "q27", 100);
tpcds_sf1_union_query_parity_test!(tpcds_sf1_union_q33_matches_as_whole_native, "q33", 100);
tpcds_sf1_union_query_parity_test!(tpcds_sf1_union_q56_matches_as_whole_native, "q56", 100);
tpcds_sf1_union_query_parity_test!(tpcds_sf1_union_q60_matches_as_whole_native, "q60", 100);
tpcds_sf1_union_query_parity_test!(tpcds_sf1_union_q66_matches_as_whole_native, "q66", 5);
tpcds_sf1_union_query_parity_test!(tpcds_sf1_union_q71_matches_as_whole_native, "q71", 1018);
tpcds_sf1_union_query_parity_test!(tpcds_sf1_union_q75_matches_as_whole_native, "q75", 100);
tpcds_sf1_union_query_parity_test!(tpcds_sf1_union_q76_matches_as_whole_native, "q76", 100);

#[tokio::test]
async fn tpcds_sf1_projection_scalar_queries_match_current_admission_contract() {
    let Some(root) = sf1_fixture_root_or_skip() else {
        return;
    };
    let root = root.as_path();
    assert!(root.join("store_sales.parquet").is_file(), "TPC-DS SF1 fixture is missing at {}", root.display());

    let query_filter = vec!["q02".to_string(), "q78".to_string(), "q84".to_string()];
    let items = tpcds_parity_workload_items(
        root,
        sf1_tpcds_test_session_config(),
        sf1_native_optimizer_config(),
        Some(&query_filter),
    )
    .await
    .unwrap();
    let report = NativeTpcdsParityReport::collect(items, ParityFloatTolerance::exact());
    assert_eq!(report.rows.len(), 3);

    for (query_id, output_rows) in [("q02", 2513), ("q78", 100), ("q84", 25)] {
        let row = report.rows.iter().find(|row| row.query_id == query_id).expect("query row should exist");
        assert_whole_native_parity_row(row, query_id, ParityComparisonMode::OrderedExact, output_rows);
    }
}

#[tokio::test]
async fn tpcds_sf1_distinct_aggregate_global_limit_query_matches_as_whole_native() {
    let Some(root) = sf1_fixture_root_or_skip() else {
        return;
    };
    let root = root.as_path();
    assert!(root.join("store_sales.parquet").is_file(), "TPC-DS SF1 fixture is missing at {}", root.display());

    let lowering = lower_physical_plan_to_native(plain_tpcds_physical_plan(root, "q28").await);
    assert_eq!(lowering.rewritten_count(), 1, "outcomes={:?}", lowering.report.outcomes());
    assert_eq!(lowering.rejected_count(), 0, "outcomes={:?}", lowering.report.outcomes());
    let optimized = nexus_query_engine::optimizer::optimize_query_plan(
        lowering.plans[0].clone(),
        &nexus_query_engine::optimizer::NativeOptimizerOptions::default(),
    )
    .plan;
    assert!(
        plan_contains_aggregate_function(&optimized, nexus_query_engine::expr::AggregateFunction::NUnique),
        "q28 must retain count-distinct as the native NUnique aggregate:\n{optimized}"
    );

    let query_filter = vec!["q28".to_string()];
    let items = tpcds_parity_workload_items(
        root,
        sf1_tpcds_test_session_config(),
        sf1_native_optimizer_config(),
        Some(&query_filter),
    )
    .await
    .unwrap();
    let report = NativeTpcdsParityReport::collect(items, ParityFloatTolerance::exact());
    assert_eq!(report.rows.len(), 1);

    let row = &report.rows[0];
    assert_whole_native_parity_row(row, "q28", ParityComparisonMode::UnorderedExact, 1);
}

#[tokio::test]
async fn tpcds_sf1_hash_join_queries_match_with_expected_native_boundaries() {
    let Some(root) = sf1_fixture_root_or_skip() else {
        return;
    };
    let root = root.as_path();
    assert!(root.join("store_sales.parquet").is_file(), "TPC-DS SF1 fixture is missing at {}", root.display());

    let query_filter = vec!["q13".to_string(), "q38".to_string(), "q87".to_string()];
    let items = tpcds_parity_workload_items(
        root,
        sf1_tpcds_test_session_config(),
        sf1_native_optimizer_config(),
        Some(&query_filter),
    )
    .await
    .unwrap();
    let report = NativeTpcdsParityReport::collect(items, ParityFloatTolerance::exact());
    assert_eq!(report.rows.len(), 3);

    let q13 = report.rows.iter().find(|row| row.query_id == "q13").expect("q13 row should exist");
    assert_whole_native_parity_row(q13, "q13", ParityComparisonMode::UnorderedExact, 1);

    let q38 = report.rows.iter().find(|row| row.query_id == "q38").expect("q38 row should exist");
    assert_whole_native_parity_row(q38, "q38", ParityComparisonMode::UnorderedExact, 1);

    let q87 = report.rows.iter().find(|row| row.query_id == "q87").expect("q87 row should exist");
    assert_whole_native_parity_row(q87, "q87", ParityComparisonMode::UnorderedExact, 1);
}

#[test]
fn tpcds_sf1_root_topk_join_aggregate_queries_match_as_whole_native() {
    run_tpcds_sf1_heavy_planning_test("tpcds_sf1_root_topk_join_aggregate", || async {
        let Some(root) = sf1_fixture_root_or_skip() else {
            return;
        };
        let root = root.as_path();
        assert!(root.join("store_sales.parquet").is_file(), "TPC-DS SF1 fixture is missing at {}", root.display());

        // q08 stays pinned by planning/admission; runtime parity exceeds this
        // Cargo target's memory contract and belongs in explicit TPC-DS reports.
        for (query_id, output_rows) in [
            ("q01", 100),
            ("q04", 8),
            ("q11", 88),
            ("q41", 4),
            ("q46", 100),
            ("q58", 3),
            ("q68", 100),
            ("q74", 92),
            ("q78", 100),
            ("q79", 100),
            ("q83", 21),
        ] {
            match query_id {
                "q04" => {
                    assert_tpcds_sf1_single_query_partial_native_parity(
                        root,
                        query_id,
                        ParityComparisonMode::OrderedExact,
                        output_rows,
                        6,
                        24,
                    )
                    .await;
                },
                "q11" => {
                    assert_tpcds_sf1_single_query_partial_native_parity(
                        root,
                        query_id,
                        ParityComparisonMode::OrderedExact,
                        output_rows,
                        4,
                        16,
                    )
                    .await;
                },
                "q41" => {
                    assert_tpcds_sf1_single_query_partial_native_parity(
                        root,
                        query_id,
                        ParityComparisonMode::OrderedExact,
                        output_rows,
                        2,
                        4,
                    )
                    .await;
                },
                "q74" => {
                    assert_tpcds_sf1_single_query_partial_native_parity(
                        root,
                        query_id,
                        ParityComparisonMode::OrderedExact,
                        output_rows,
                        4,
                        16,
                    )
                    .await;
                },
                "q83" => {
                    assert_tpcds_sf1_single_query_partial_native_parity(
                        root,
                        query_id,
                        ParityComparisonMode::OrderedExact,
                        output_rows,
                        3,
                        11,
                    )
                    .await;
                },
                _ => {
                    assert_tpcds_sf1_single_query_whole_native_parity(
                        root,
                        query_id,
                        ParityComparisonMode::OrderedExact,
                        output_rows,
                    )
                    .await;
                },
            }
        }
    });
}

#[test]
fn tpcds_sf1_root_topk_shared_fanout_queries_match_as_whole_native() {
    run_tpcds_sf1_heavy_planning_test("tpcds_sf1_root_topk_shared_fanout", || async {
        let Some(root) = sf1_fixture_root_or_skip() else {
            return;
        };
        let root = root.as_path();
        assert!(root.join("store_sales.parquet").is_file(), "TPC-DS SF1 fixture is missing at {}", root.display());

        let query_filter = vec!["q30".to_string(), "q59".to_string(), "q65".to_string(), "q81".to_string()];
        let items = tpcds_parity_workload_items(
            root,
            sf1_tpcds_test_session_config(),
            sf1_native_optimizer_config(),
            Some(&query_filter),
        )
        .await
        .unwrap();
        let report = NativeTpcdsParityReport::collect(items, ParityFloatTolerance::exact());
        assert_eq!(report.rows.len(), 4);

        for query_id in ["q30", "q65", "q81"] {
            let row = report.rows.iter().find(|row| row.query_id == query_id).expect("query row should exist");
            assert_whole_native_parity_row(row, query_id, ParityComparisonMode::OrderedExact, 100);
        }
        let q59 = report.rows.iter().find(|row| row.query_id == "q59").expect("q59 row should exist");
        assert_partial_native_parity_row(q59, "q59", ParityComparisonMode::OrderedExact, 100, 2, 9);
    });
}

#[tokio::test]
async fn tpcds_sf1_outer_hash_join_query_matches_as_whole_native() {
    let Some(root) = sf1_fixture_root_or_skip() else {
        return;
    };
    let root = root.as_path();
    assert!(root.join("catalog_sales.parquet").is_file(), "TPC-DS SF1 fixture is missing at {}", root.display());

    let query_filter = vec!["q72".to_string()];
    let items = tpcds_parity_workload_items(
        root,
        sf1_tpcds_test_session_config(),
        q72_native_optimizer_config(),
        Some(&query_filter),
    )
    .await
    .unwrap();
    let report = NativeTpcdsParityReport::collect(items, ParityFloatTolerance::exact());
    assert_eq!(report.rows.len(), 1);

    let row = &report.rows[0];
    assert_eq!(row.query_id, "q72");
    assert_eq!(row.cpu_planning_status, ParityStageStatus::Succeeded);
    assert_eq!(row.cpu_runtime_status, ParityStageStatus::Succeeded);
    assert_eq!(row.native_planning_outcome, NativeTpcdsParityPlanningOutcome::AcceptedReplaced);
    assert_eq!(row.native_reason_code, None);
    assert_eq!(row.native_unsupported_category, None);
    assert_eq!(row.post_native_execution_path, NativeDataFusionPostNativeExecutionPath::Native);
    assert_eq!(row.native_plan_fragment_count, 1);
    assert_eq!(row.host_datafusion_boundary_count, 0);
    assert!(row.native_runtime_started, "q72 must execute as one whole native subtree");
    assert_eq!(row.native_runtime_outcome, NativeDataFusionRuntimeOutcome::Succeeded);
    assert_eq!(row.parity_status, ParityStatus::Matched);
    assert_eq!(row.comparison_mode, ParityComparisonMode::OrderedExact);
    assert_eq!(row.cpu_output_rows, Some(100));
    assert_eq!(row.native_output_rows, Some(100));
    assert_eq!(row.native_reservation_denied_count, Some(0));
    assert_eq!(row.native_retry_count, Some(0));
    assert_eq!(row.native_runtime_failure, None);
}

#[tokio::test]
async fn tpcds_sf1_q72_outer_hash_join_lowering_admits_whole_native_plan() {
    let Some(root) = sf1_fixture_root_or_skip() else {
        return;
    };
    let root = root.as_path();
    assert!(root.join("catalog_sales.parquet").is_file(), "TPC-DS SF1 fixture is missing at {}", root.display());

    let lowering = lower_physical_plan_to_native(plain_tpcds_physical_plan(root, "q72").await);
    assert_eq!(lowering.rewritten_count(), 1, "q72 outcomes={:?}", lowering.report.outcomes());
    assert_eq!(lowering.rejected_count(), 0, "q72 outcomes={:?}", lowering.report.outcomes());

    let optimized = nexus_query_engine::optimizer::optimize_query_plan(
        lowering.plans[0].clone(),
        &nexus_query_engine::optimizer::NativeOptimizerOptions::default(),
    )
    .plan;
    let rendered = optimized.to_string();
    assert!(rendered.contains("type=left"), "q72 should lower left/right outer joins natively:\n{rendered}");

    let analysis = optimized.capability_analysis();
    let admission = nexus_query_engine::admission::strict_bounded_pipeline_admission(&optimized)
        .expect("strict admission should not fail");
    assert!(
        admission.is_some(),
        "q72 strict bounded admission should accept the whole native plan; first_rejection={:?}\n{rendered}",
        analysis.first_rejection_projection()
    );
}

#[tokio::test]
async fn tpcds_sf1_precision_sensitive_stddev_matches_whole_native_exact() {
    let Some(root) = sf1_fixture_root_or_skip() else {
        return;
    };
    let root = root.as_path();
    assert!(root.join("store_sales.parquet").is_file(), "TPC-DS SF1 fixture is missing at {}", root.display());

    let query_filter = vec!["q39".to_string()];
    // q39 pins an exact floating statistical aggregate contract; changing CPU
    // partitions changes DataFusion's merge order and invalidates exact parity.
    let items = tpcds_parity_workload_items(
        root,
        tpcds_report_session_config(),
        sf1_native_optimizer_config(),
        Some(&query_filter),
    )
    .await
    .unwrap();
    let report = NativeTpcdsParityReport::collect(items, ParityFloatTolerance::exact());
    assert_eq!(report.rows.len(), 1);

    let q39 = report.rows.iter().find(|row| row.query_id == "q39").expect("q39 row should exist");
    assert_whole_native_parity_row(q39, "q39", ParityComparisonMode::OrderedExact, 243);
}

#[tokio::test]
async fn tpcds_sf1_q39_tolerant_mode_matches_with_float_contract_without_exact_host_gates() {
    let Some(root) = sf1_fixture_root_or_skip() else {
        return;
    };
    let root = root.as_path();
    assert!(root.join("store_sales.parquet").is_file(), "TPC-DS SF1 fixture is missing at {}", root.display());

    let config = sf1_native_optimizer_config()
        .with_statistical_aggregate_execution_mode(StatisticalAggregateExecutionMode::GpuNativeTolerant);
    let rule = CudfNativePhysicalOptimizerRule::new(config.clone());
    let plan = plain_tpcds_physical_plan(root, "q39").await;
    let optimized = rule.optimize(plan, &ConfigOptions::default()).unwrap();
    let report = NativeDataFusionPlanningReport::from_retained_rule("q39", optimized, &rule);
    assert_eq!(report.rows.len(), 1);

    let q39 = &report.rows[0];
    assert_eq!(q39.outcome, NativeDataFusionPlanningOutcome::AcceptedReplaced);
    assert_eq!(q39.reason_code, None);
    assert_eq!(q39.post_native_execution_path, NativeDataFusionPostNativeExecutionPath::Native);
    assert_eq!(q39.native_plan_fragment_count, 1);
    assert_eq!(q39.host_datafusion_boundary_count, 0);
    assert!(!q39.rejected_before_execution);
    assert_eq!(q39.strict_bounded_compiled_graphs.len(), 1);
    assert!(q39.strict_bounded_contract_rows.iter().any(|gate| {
        gate.required_contract == "gpu_native_tolerant_statistical_aggregate_contract"
            && gate.memory_boundary == "tolerant_statistical_join_final_full_sort"
            && gate.contract_status.as_str() == "modeled"
    }));
    assert!(q39.strict_bounded_contract_rows.iter().any(|gate| {
        gate.required_contract == "final_full_sort_output_cardinality_bound"
            && gate.contract_status.as_str() == "modeled"
    }));
    assert!(
        q39.strict_bounded_contract_rows.iter().all(|gate| {
            !matches!(
                gate.required_contract,
                "host_welford_input_materialization_preflight" | "host_welford_output_upload_reservation"
            )
        }),
        "tolerant mode must not publish exact host-Welford gates"
    );

    let query_filter = vec!["q39".to_string()];
    let items =
        tpcds_parity_workload_items(root, tpcds_report_session_config(), config, Some(&query_filter)).await.unwrap();
    let report = NativeTpcdsParityReport::collect(items, ParityFloatTolerance::new(1e-8, 1e-8));
    assert_eq!(report.rows.len(), 1);
    let q39 = report.rows.iter().find(|row| row.query_id == "q39").expect("q39 row should exist");
    assert_whole_native_parity_row(q39, "q39", ParityComparisonMode::OrderedFloatTolerant, 243);
    let tsv =
        report.with_statistical_aggregate_execution_mode(StatisticalAggregateExecutionMode::GpuNativeTolerant).to_tsv();
    assert!(tsv.contains("# statistical_aggregate_execution_mode=gpu_native_tolerant\n"));
    assert!(tsv.contains("# statistical_aggregate_tolerance_contract=float_tolerant\n"));
    assert!(tsv.contains("# parity_float_tolerance_absolute=0.00000001\n"));
    assert!(tsv.contains("# parity_float_tolerance_relative=0.00000001\n"));
    assert!(
        tsv.contains("\tordered_float_tolerant\t"),
        "q39 tolerant parity row must declare ordered float-tolerant comparison:\n{tsv}"
    );
}

#[tokio::test]
async fn tpcds_sf1_q39_adaptive_mode_admits_with_adaptive_pipeline_report() {
    let Some(root) = sf1_fixture_root_or_skip() else {
        return;
    };
    let root = root.as_path();
    assert!(root.join("store_sales.parquet").is_file(), "TPC-DS SF1 fixture is missing at {}", root.display());

    let config = sf1_native_optimizer_config()
        .with_native_pipeline_execution_mode(NativePipelineExecutionMode::Adaptive)
        .with_statistical_aggregate_execution_mode(StatisticalAggregateExecutionMode::GpuNativeTolerant);
    let rule = CudfNativePhysicalOptimizerRule::new(config);
    let plan = plain_tpcds_physical_plan(root, "q39").await;
    let optimized = rule.optimize(plan, &ConfigOptions::default()).unwrap();
    let report = NativeDataFusionPlanningReport::from_retained_rule("q39", optimized, &rule);
    assert_eq!(report.rows.len(), 1);

    let q39 = &report.rows[0];
    assert_eq!(q39.outcome, NativeDataFusionPlanningOutcome::AcceptedReplaced);
    assert_eq!(q39.reason_code, None);
    assert_eq!(q39.post_native_execution_path, NativeDataFusionPostNativeExecutionPath::Native);
    assert_eq!(q39.native_plan_fragment_count, 1);
    assert_eq!(q39.host_datafusion_boundary_count, 0);
    assert!(!q39.rejected_before_execution);
    assert!(q39.strict_bounded_contract_rows.is_empty());
    assert!(q39.strict_bounded_compiled_graphs.is_empty());
    assert_eq!(q39.adaptive_pipeline_rows.len(), 1);
    let adaptive = q39.adaptive_pipeline_rows[0].to_stable_kv_row();
    assert!(adaptive.contains("execution_mode=adaptive"), "{adaptive}");
    assert!(adaptive.contains("candidate_status=admitted"), "{adaptive}");
    assert!(adaptive.contains("adaptive_strategy=conservative_compiled_pipeline"), "{adaptive}");
    assert!(adaptive.contains("rejection_reason=none"), "{adaptive}");
    assert!(!adaptive.contains("execution_mode=strict_bounded"), "{adaptive}");
    assert!(
        !adaptive.contains("adaptive_exact_statistical_join_final_full_sort_requires_explicit_contract"),
        "{adaptive}"
    );
}

#[tokio::test]
async fn tpcds_sf1_cross_join_count_subplans_match_after_hidden_join_output_projection() {
    let Some(root) = sf1_fixture_root_or_skip() else {
        return;
    };
    let root = root.as_path();
    assert!(root.join("web_sales.parquet").is_file(), "TPC-DS SF1 fixture is missing at {}", root.display());

    let query_filter = vec!["q88".to_string(), "q90".to_string()];
    let items = tpcds_parity_workload_items(
        root,
        sf1_tpcds_test_session_config(),
        sf1_native_optimizer_config(),
        Some(&query_filter),
    )
    .await
    .unwrap();
    let report = NativeTpcdsParityReport::collect(items, ParityFloatTolerance::exact());
    assert_eq!(report.rows.len(), 2);

    for (query_id, comparison_mode) in
        [("q88", ParityComparisonMode::UnorderedExact), ("q90", ParityComparisonMode::OrderedExact)]
    {
        let row = report.rows.iter().find(|row| row.query_id == query_id).expect("query row should exist");
        assert_whole_native_parity_row(row, query_id, comparison_mode, 1);
    }
}
