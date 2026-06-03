use std::collections::BTreeSet;

use datafusion_nexus::{
    NativeDataFusionPlanningReport, NativeDataFusionPostNativeExecutionPath, NativeDataFusionRuntimeOutcome,
    NativeTpcdsParityPlanningOutcome, NativeTpcdsParityReport, ParityComparisonMode, ParityFloatTolerance,
    ParityStageStatus, ParityStatus, lower_physical_plan_to_native,
};
use datafusion_nexus_bench::{
    tpcds_datafusion_planning_workload_items_with_optimizer_config, tpcds_report_session_config,
};

use super::support::*;

#[tokio::test]
async fn tpcds_sf1_q06_matches_after_hash_join_reservation_contract_fix() {
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
    assert_eq!(row.query_id, "q06");
    assert_eq!(row.cpu_planning_status, ParityStageStatus::Succeeded);
    assert_eq!(row.cpu_runtime_status, ParityStageStatus::Succeeded);
    assert_eq!(row.native_planning_outcome, NativeTpcdsParityPlanningOutcome::AcceptedReplaced);
    assert_eq!(row.native_reason_code, None);
    assert_eq!(row.native_unsupported_category, None);
    assert!(row.native_runtime_started, "q06 must execute through the native path");
    assert_eq!(row.native_runtime_outcome, NativeDataFusionRuntimeOutcome::Succeeded);
    assert_eq!(row.parity_status, ParityStatus::Matched);
    assert_ne!(row.parity_status, ParityStatus::NativeRejected);
    assert_eq!(row.comparison_mode, ParityComparisonMode::OrderedExact);
    assert_eq!(row.cpu_output_rows, Some(46));
    assert_eq!(row.native_output_rows, Some(46));
    assert_eq!(row.native_reservation_denied_count, Some(0));
    assert_eq!(row.native_retry_count, Some(0));
    assert_eq!(row.native_runtime_failure, None);
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
    assert_eq!(row.query_id, "q85");
    assert_eq!(row.cpu_planning_status, ParityStageStatus::Succeeded);
    assert_eq!(row.cpu_runtime_status, ParityStageStatus::Succeeded);
    assert_eq!(row.native_planning_outcome, NativeTpcdsParityPlanningOutcome::AcceptedReplaced);
    assert_eq!(row.native_reason_code, None);
    assert_eq!(row.native_unsupported_category, None);
    assert!(row.native_runtime_started, "q85 must execute through the native path");
    assert_eq!(row.native_runtime_outcome, NativeDataFusionRuntimeOutcome::Succeeded);
    assert_eq!(row.parity_status, ParityStatus::Matched);
    assert_eq!(row.comparison_mode, ParityComparisonMode::OrderedExact);
    assert_eq!(row.cpu_output_rows, Some(1));
    assert_eq!(row.native_output_rows, Some(1));
    assert_eq!(row.native_reservation_denied_count, Some(0));
    assert_eq!(row.native_retry_count, Some(0));
    assert_eq!(row.native_runtime_failure, None);
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

    for (query_id, output_rows) in [("q04", 6), ("q11", 90), ("q31", 44), ("q74", 92)] {
        let row = report.rows.iter().find(|row| row.query_id == query_id).expect("query row should exist");
        assert_eq!(row.cpu_planning_status, ParityStageStatus::Succeeded, "{query_id}");
        assert_eq!(row.cpu_runtime_status, ParityStageStatus::Succeeded, "{query_id}");
        assert_eq!(row.native_planning_outcome, NativeTpcdsParityPlanningOutcome::AcceptedReplaced, "{query_id}");
        assert_eq!(row.native_reason_code, None, "{query_id}");
        assert_eq!(row.native_unsupported_category, None, "{query_id}");
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
        assert_eq!(row.cpu_planning_status, ParityStageStatus::Succeeded, "{query_id}");
        assert_eq!(row.cpu_runtime_status, ParityStageStatus::Succeeded, "{query_id}");
        assert_eq!(row.native_planning_outcome, NativeTpcdsParityPlanningOutcome::AcceptedReplaced, "{query_id}");
        assert_eq!(row.native_reason_code, None, "{query_id}");
        assert_eq!(row.native_unsupported_category, None, "{query_id}");
        assert!(row.native_runtime_started, "{query_id} must execute through the native path");
        assert_eq!(row.native_runtime_outcome, NativeDataFusionRuntimeOutcome::Succeeded, "{query_id}");
        assert_eq!(row.parity_status, ParityStatus::Matched, "{query_id}");
        assert_eq!(row.comparison_mode, comparison_mode, "{query_id}");
        assert_eq!(row.cpu_output_rows, Some(1), "{query_id}");
        assert_eq!(row.native_output_rows, Some(1), "{query_id}");
        assert_eq!(row.native_reservation_denied_count, Some(0), "{query_id}");
        assert_eq!(row.native_retry_count, Some(0), "{query_id}");
        assert_eq!(row.native_runtime_failure, None, "{query_id}");
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
    assert_eq!(row.query_id, "q61");
    assert_eq!(row.cpu_planning_status, ParityStageStatus::Succeeded);
    assert_eq!(row.cpu_runtime_status, ParityStageStatus::Succeeded);
    assert_eq!(row.native_planning_outcome, NativeTpcdsParityPlanningOutcome::AcceptedReplaced);
    assert_eq!(row.native_reason_code, None);
    assert_eq!(row.native_unsupported_category, None);
    assert!(row.native_runtime_started, "q61 must execute through the native path");
    assert_eq!(row.native_runtime_outcome, NativeDataFusionRuntimeOutcome::Succeeded);
    assert_eq!(row.parity_status, ParityStatus::Matched);
    assert_eq!(row.comparison_mode, ParityComparisonMode::OrderedExact);
    assert_eq!(row.cpu_output_rows, Some(1));
    assert_eq!(row.native_output_rows, Some(1));
    assert_eq!(row.native_reservation_denied_count, Some(0));
    assert_eq!(row.native_retry_count, Some(0));
    assert_eq!(row.native_runtime_failure, None);
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
    assert_eq!(row.query_id, "q09");
    assert_eq!(row.cpu_planning_status, ParityStageStatus::Succeeded);
    assert_eq!(row.cpu_runtime_status, ParityStageStatus::Succeeded);
    assert_eq!(row.native_planning_outcome, NativeTpcdsParityPlanningOutcome::AcceptedReplaced);
    assert_eq!(row.native_reason_code, None);
    assert_eq!(row.native_unsupported_category, None);
    assert!(row.native_runtime_started, "q09 must execute through the native path");
    assert_eq!(row.native_runtime_outcome, NativeDataFusionRuntimeOutcome::Succeeded);
    assert_eq!(row.parity_status, ParityStatus::Matched);
    assert_eq!(row.comparison_mode, ParityComparisonMode::UnorderedExact);
    assert_eq!(row.cpu_output_rows, Some(1));
    assert_eq!(row.native_output_rows, Some(1));
    assert_eq!(row.native_reservation_denied_count, Some(0));
    assert_eq!(row.native_retry_count, Some(0));
    assert_eq!(row.native_runtime_failure, None);
}

#[tokio::test]
async fn tpcds_sf1_current_partial_native_coverage_set_is_pinned() {
    let Some(root) = sf1_fixture_root_or_skip() else {
        return;
    };
    let root = root.as_path();
    assert!(root.join("store_sales.parquet").is_file(), "TPC-DS SF1 fixture is missing at {}", root.display());

    let items = tpcds_datafusion_planning_workload_items_with_optimizer_config(
        root,
        tpcds_report_session_config(),
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
    let expected: BTreeSet<&str> = BTreeSet::from([
        "q01", "q04", "q06", "q08", "q09", "q11", "q24", "q28", "q30", "q31", "q32", "q34", "q38", "q41", "q44", "q46",
        "q47", "q51", "q57", "q58", "q59", "q61", "q64", "q65", "q67", "q68", "q73", "q74", "q78", "q79", "q81", "q83",
        "q87", "q88", "q90", "q92", "q97",
    ]);
    let cpu_expected: BTreeSet<&str> = BTreeSet::from(["q39", "q70", "q86"]);

    assert_eq!(partial_native, expected);
    assert_eq!(cpu_only, cpu_expected);
    for query_id in expected {
        let rows: Vec<_> = report.rows.iter().filter(|row| row.query_id == query_id).collect();
        assert!(!rows.is_empty(), "{query_id} should have planning rows");
        assert!(rows.iter().all(|row| row.native_plan_fragment_count > 0), "{query_id} should report native fragments");
        assert!(
            rows.iter().all(|row| row.host_datafusion_boundary_count > 0),
            "{query_id} should report host DataFusion boundary operators"
        );
    }
    for query_id in ["q39", "q70", "q86"] {
        let row =
            report.rows.iter().find(|row| row.query_id == query_id).unwrap_or_else(|| panic!("{query_id} is missing"));
        assert_strict_bounded_rejected_before_execution_planning_row(row, query_id, "operator_hashjoin");
    }
}

#[tokio::test]
async fn tpcds_sf1_q44_keyless_outer_cross_join_matches_with_partial_native_children() {
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
    assert_partial_native_parity_row(row, "q44", ParityComparisonMode::OrderedExact, 10);
}

#[tokio::test]
async fn tpcds_sf1_bitwise_grouping_window_queries_match_current_admission_contract() {
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
    assert_rejected_before_native_execution_parity_row(q70, "q70", "operator_hashjoin", 3);

    let q86 = report.rows.iter().find(|row| row.query_id == "q86").expect("q86 row should exist");
    assert_rejected_before_native_execution_parity_row(q86, "q86", "operator_hashjoin", 100);
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

    for query_id in ["q47", "q57", "q67"] {
        let row = report.rows.iter().find(|row| row.query_id == query_id).expect("query row should exist");
        assert_partial_native_parity_row(row, query_id, ParityComparisonMode::OrderedExact, 100);
    }
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

    for (query_id, output_rows) in [("q12", 100), ("q20", 100), ("q98", 2521)] {
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
async fn tpcds_sf1_cumulative_aggregate_window_query_matches_with_partial_native_children() {
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
    assert_partial_native_parity_row(row, query_id, ParityComparisonMode::OrderedExact, output_rows);
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
    assert_eq!(row.cpu_output_rows, Some(34));
    assert_eq!(row.native_output_rows, Some(34));
    assert_eq!(row.native_reservation_denied_count, Some(0));
    assert_eq!(row.native_retry_count, Some(0));
    assert_eq!(row.native_runtime_failure, None);
}

#[tokio::test]
async fn tpcds_sf1_round_decimal_segment_query_matches_as_whole_native() {
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
    assert_eq!(row.cpu_planning_status, ParityStageStatus::Succeeded);
    assert_eq!(row.cpu_runtime_status, ParityStageStatus::Succeeded);
    assert_eq!(row.native_planning_outcome, NativeTpcdsParityPlanningOutcome::AcceptedReplaced);
    assert_eq!(row.post_native_execution_path, NativeDataFusionPostNativeExecutionPath::Native);
    assert_eq!(row.native_plan_fragment_count, 1);
    assert_eq!(row.host_datafusion_boundary_count, 0);
    assert!(row.native_runtime_started, "q54 must execute through the native path");
    assert_eq!(row.native_runtime_outcome, NativeDataFusionRuntimeOutcome::Succeeded);
    assert_eq!(row.parity_status, ParityStatus::Matched);
    assert_eq!(row.comparison_mode, ParityComparisonMode::OrderedExact);
    assert_eq!(row.cpu_output_rows, Some(1));
    assert_eq!(row.native_output_rows, Some(1));
    assert_eq!(row.native_reservation_denied_count, Some(0));
    assert_eq!(row.native_retry_count, Some(0));
    assert_eq!(row.native_runtime_failure, None);
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

    for (query_id, output_rows) in [("q10", 6), ("q35", 100), ("q45", 19)] {
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

    let (query_id, output_rows) = ("q17", 0);
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

        let optimized = nexus_query_engine::optimize_query_plan(
            lowering.plans[0].clone(),
            &nexus_query_engine::NativeOptimizerOptions::default(),
        )
        .plan;
        let rendered = optimized.to_string();
        assert!(rendered.contains("Aggregate"), "{query_id} must retain native aggregate lowering:\n{rendered}");
        let admission = nexus_query_engine::strict_bounded_pipeline_admission(&optimized)
            .expect("strict admission should not fail");
        assert!(admission.is_some(), "{query_id} strict bounded admission should accept the whole native plan");

        if query_id == "q05" {
            assert!(rendered.contains("CONCAT("), "q05 should lower scalar concat into native projection:\n{rendered}");
        }
        if query_id == "q18" {
            assert!(rendered.contains("Avg("), "q18 grouping-set AVG state should lower natively:\n{rendered}");
            assert!(
                rendered.contains("__cudf_aggregate_arg_"),
                "q18 AVG inputs should be materialized as reusable native aggregate arguments:\n{rendered}"
            );
        }
    }
}

#[tokio::test]
async fn tpcds_sf1_union_queries_match_as_whole_native() {
    let Some(root) = sf1_fixture_root_or_skip() else {
        return;
    };
    let root = root.as_path();
    assert!(root.join("store_sales.parquet").is_file(), "TPC-DS SF1 fixture is missing at {}", root.display());

    let query_filter = vec![
        "q23".to_string(),
        "q27".to_string(),
        "q33".to_string(),
        "q71".to_string(),
        "q75".to_string(),
        "q56".to_string(),
        "q60".to_string(),
        "q66".to_string(),
        "q76".to_string(),
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
    assert_eq!(report.rows.len(), 9);

    for (query_id, output_rows) in [
        ("q23", 4),
        ("q27", 100),
        ("q33", 100),
        ("q56", 100),
        ("q60", 100),
        ("q66", 5),
        ("q71", 1031),
        ("q75", 100),
        ("q76", 100),
    ] {
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

    for (query_id, output_rows) in [("q02", 2513), ("q84", 16)] {
        let row = report.rows.iter().find(|row| row.query_id == query_id).expect("query row should exist");
        assert_whole_native_parity_row(row, query_id, ParityComparisonMode::OrderedExact, output_rows);
    }
    let q78 = report.rows.iter().find(|row| row.query_id == "q78").expect("q78 row should exist");
    assert_partial_native_parity_row(q78, "q78", ParityComparisonMode::OrderedExact, 100);
}

#[tokio::test]
async fn tpcds_sf1_distinct_aggregate_global_limit_query_matches_with_partial_native_children() {
    let Some(root) = sf1_fixture_root_or_skip() else {
        return;
    };
    let root = root.as_path();
    assert!(root.join("store_sales.parquet").is_file(), "TPC-DS SF1 fixture is missing at {}", root.display());

    let lowering = lower_physical_plan_to_native(plain_tpcds_physical_plan(root, "q28").await);
    assert_eq!(lowering.rewritten_count(), 1, "outcomes={:?}", lowering.report.outcomes());
    assert_eq!(lowering.rejected_count(), 0, "outcomes={:?}", lowering.report.outcomes());
    let optimized = nexus_query_engine::optimize_query_plan(
        lowering.plans[0].clone(),
        &nexus_query_engine::NativeOptimizerOptions::default(),
    )
    .plan;
    assert!(
        plan_contains_aggregate_function(&optimized, nexus_query_engine::AggregateFunction::NUnique),
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
    assert_partial_native_parity_row(row, "q28", ParityComparisonMode::UnorderedExact, 1);
}

#[tokio::test]
async fn tpcds_sf1_hash_join_queries_match_with_partial_native_children() {
    let Some(root) = sf1_fixture_root_or_skip() else {
        return;
    };
    let root = root.as_path();
    assert!(root.join("store_sales.parquet").is_file(), "TPC-DS SF1 fixture is missing at {}", root.display());

    let query_filter = vec![
        "q01".to_string(),
        "q30".to_string(),
        "q38".to_string(),
        "q65".to_string(),
        "q81".to_string(),
        "q87".to_string(),
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

    for (query_id, comparison_mode, output_rows) in [
        ("q01", ParityComparisonMode::OrderedExact, 100),
        ("q30", ParityComparisonMode::OrderedExact, 100),
        ("q38", ParityComparisonMode::UnorderedExact, 1),
        ("q65", ParityComparisonMode::OrderedExact, 100),
        ("q81", ParityComparisonMode::OrderedExact, 100),
        ("q87", ParityComparisonMode::UnorderedExact, 1),
    ] {
        let row = report.rows.iter().find(|row| row.query_id == query_id).expect("query row should exist");
        assert_partial_native_parity_row(row, query_id, comparison_mode, output_rows);
    }
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

    let optimized = nexus_query_engine::optimize_query_plan(
        lowering.plans[0].clone(),
        &nexus_query_engine::NativeOptimizerOptions::default(),
    )
    .plan;
    let rendered = optimized.to_string();
    assert!(rendered.contains("type=left"), "q72 should lower left/right outer joins natively:\n{rendered}");

    let analysis = optimized.capability_analysis();
    let admission =
        nexus_query_engine::strict_bounded_pipeline_admission(&optimized).expect("strict admission should not fail");
    assert!(
        admission.is_some(),
        "q72 strict bounded admission should accept the whole native plan; first_rejection={:?}\n{rendered}",
        analysis.first_rejection_projection()
    );
}

#[tokio::test]
async fn tpcds_sf1_precision_sensitive_stddev_falls_back_before_native_execution() {
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
    assert_rejected_before_native_execution_parity_row(q39, "q39", "operator_hashjoin", 243);
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
        assert_eq!(row.cpu_planning_status, ParityStageStatus::Succeeded, "{query_id}");
        assert_eq!(row.cpu_runtime_status, ParityStageStatus::Succeeded, "{query_id}");
        assert_eq!(row.native_planning_outcome, NativeTpcdsParityPlanningOutcome::AcceptedReplaced, "{query_id}");
        assert_eq!(row.native_reason_code, None, "{query_id}");
        assert_eq!(row.native_unsupported_category, None, "{query_id}");
        assert!(row.native_runtime_started, "{query_id} must execute through the native path");
        assert_eq!(row.native_runtime_outcome, NativeDataFusionRuntimeOutcome::Succeeded, "{query_id}");
        assert_eq!(row.parity_status, ParityStatus::Matched, "{query_id}");
        assert_eq!(row.comparison_mode, comparison_mode, "{query_id}");
        assert_eq!(row.cpu_output_rows, Some(1), "{query_id}");
        assert_eq!(row.native_output_rows, Some(1), "{query_id}");
        assert_eq!(row.native_reservation_denied_count, Some(0), "{query_id}");
        assert_eq!(row.native_retry_count, Some(0), "{query_id}");
        assert_eq!(row.native_runtime_failure, None, "{query_id}");
    }
}
