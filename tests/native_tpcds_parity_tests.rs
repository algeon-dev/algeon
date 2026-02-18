use std::{path::Path, sync::Arc};

use arrow::{
    array::{Decimal128Array, Float64Array, Int64Array, StringArray},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use datafusion::{
    physical_plan::ExecutionPlan,
    prelude::{ParquetReadOptions, SessionContext},
};
use datafusion_cudf::{
    CudfOptimizerConfig, GpuFallbackPolicy, NATIVE_TPCDS_PARITY_TSV_HEADER, NATIVE_TPCDS_PARITY_TSV_SCHEMA_VERSION,
    NativeCudfExecutionConfig, NativeDataFusionRuntimeOutcome, NativeTpcdsParityCpuOutcome,
    NativeTpcdsParityExternalOutput, NativeTpcdsParityExternalOutputSummary, NativeTpcdsParityNativeRuntimeOutcome,
    NativeTpcdsParityPlanningOutcome, NativeTpcdsParityReport, NativeTpcdsParityRow, NativeTpcdsParityWorkloadItem,
    ParityComparisonMode, ParityFloatTolerance, ParityStageStatus, ParityStatus, TPCDS_QUERIES, TPCDS_TABLES,
    compare_record_batches, discover_tpcds_table_layout, lower_physical_plan_to_native, sql_has_top_level_order_by,
    tpcds_parity_workload_items, tpcds_report_session_config,
};

const TPCDS_SF1_FIXTURE: &str = "/home/ubuntu/code/datafusion-cudf/tests/data/tpcds/sf1/parquet";

#[test]
fn parity_tsv_header_pins_schema_version_and_columns() {
    assert_eq!(NATIVE_TPCDS_PARITY_TSV_SCHEMA_VERSION, 2);
    let columns: Vec<&str> = NATIVE_TPCDS_PARITY_TSV_HEADER.split('\t').collect();
    assert_eq!(
        columns,
        vec![
            "query_id",
            "cpu_planning_status",
            "cpu_runtime_status",
            "native_planning_outcome",
            "native_reason_code",
            "native_unsupported_category",
            "native_runtime_started",
            "native_runtime_outcome",
            "parity_status",
            "parity_detail",
            "cpu_output_rows",
            "native_output_rows",
            "cpu_output_batches",
            "native_output_batches",
            "cpu_output_fingerprint",
            "native_output_fingerprint",
            "comparison_mode",
            "duckdb_output_status",
            "duckdb_cpu_parity_status",
            "duckdb_cpu_parity_detail",
            "duckdb_output_rows",
            "duckdb_output_batches",
            "duckdb_output_fingerprint",
            "external-engine_output_status",
            "external-engine_cpu_parity_status",
            "external-engine_cpu_parity_detail",
            "external-engine_output_rows",
            "external-engine_output_batches",
            "external-engine_output_fingerprint",
            "native_reservation_live_peak_bytes",
            "native_reservation_denied_count",
            "native_retry_count",
            "native_runtime_failure",
        ]
    );
}

#[test]
fn parity_report_to_tsv_emits_schema_version_and_header() {
    let row = NativeTpcdsParityRow {
        query_id: "q01".to_string(),
        cpu_planning_status: datafusion_cudf::ParityStageStatus::Succeeded,
        cpu_runtime_status: datafusion_cudf::ParityStageStatus::Succeeded,
        native_planning_outcome: datafusion_cudf::NativeTpcdsParityPlanningOutcome::RejectedBeforeExecution,
        native_reason_code: Some("unsupported_operator".to_string()),
        native_unsupported_category: Some("operator_join".to_string()),
        native_runtime_started: false,
        native_runtime_outcome: datafusion_cudf::NativeDataFusionRuntimeOutcome::NotStarted,
        parity_status: ParityStatus::NativeRejected,
        parity_detail: None,
        cpu_output_rows: Some(0),
        native_output_rows: None,
        cpu_output_batches: Some(0),
        native_output_batches: None,
        cpu_output_fingerprint: None,
        native_output_fingerprint: None,
        comparison_mode: ParityComparisonMode::NotApplicable,
        duckdb_output: NativeTpcdsParityExternalOutputSummary::not_configured(),
        external-engine_output: NativeTpcdsParityExternalOutputSummary::not_configured(),
        native_reservation_live_peak_bytes: None,
        native_reservation_denied_count: None,
        native_retry_count: None,
        native_runtime_failure: None,
    };
    let report = NativeTpcdsParityReport::from_rows(vec![row], ParityFloatTolerance::exact());
    let tsv = report.to_tsv();
    let first_line = tsv.lines().next().unwrap();
    assert_eq!(
        first_line,
        format!("# native_tpcds_parity_tsv_schema_version={NATIVE_TPCDS_PARITY_TSV_SCHEMA_VERSION}")
    );
    assert!(tsv.contains(NATIVE_TPCDS_PARITY_TSV_HEADER));
    // 33 columns separated by tabs in the data row.
    let data_line = tsv.lines().find(|line| line.starts_with("q01\t")).expect("data row should exist");
    assert_eq!(data_line.split('\t').count(), 33);
}

#[tokio::test]
async fn tpcds_sf1_q06_matches_after_hash_join_reservation_contract_fix() {
    let root = Path::new(TPCDS_SF1_FIXTURE);
    assert!(root.join("store_sales.parquet").is_file(), "TPC-DS SF1 fixture is missing at {}", root.display());

    let query_filter = vec!["q06".to_string()];
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
    let root = Path::new(TPCDS_SF1_FIXTURE);
    assert!(root.join("web_sales.parquet").is_file(), "TPC-DS SF1 fixture is missing at {}", root.display());

    let plain_plan = plain_tpcds_physical_plan(root, "q85").await;
    let lowering = lower_physical_plan_to_native(plain_plan);
    assert!(lowering.rewritten_count() > 0, "q85 should still lower against a plain DataFusion physical plan");

    let query_filter = vec!["q85".to_string()];
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
async fn tpcds_sf1_case_residual_post_filter_queries_match_after_one_shot_admission() {
    let root = Path::new(TPCDS_SF1_FIXTURE);
    assert!(root.join("customer.parquet").is_file(), "TPC-DS SF1 fixture is missing at {}", root.display());

    let query_filter = vec!["q04".to_string(), "q11".to_string(), "q31".to_string(), "q74".to_string()];
    let items = tpcds_parity_workload_items(
        root,
        tpcds_report_session_config(),
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
    let root = Path::new(TPCDS_SF1_FIXTURE);
    assert!(root.join("catalog_sales.parquet").is_file(), "TPC-DS SF1 fixture is missing at {}", root.display());

    let query_filter = vec!["q15".to_string()];
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
    let root = Path::new(TPCDS_SF1_FIXTURE);
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
        tpcds_report_session_config(),
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
    let root = Path::new(TPCDS_SF1_FIXTURE);
    assert!(root.join("store_sales.parquet").is_file(), "TPC-DS SF1 fixture is missing at {}", root.display());

    let query_filter = vec!["q61".to_string()];
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
    let root = Path::new(TPCDS_SF1_FIXTURE);
    assert!(root.join("store_sales.parquet").is_file(), "TPC-DS SF1 fixture is missing at {}", root.display());

    let query_filter = vec!["q09".to_string()];
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
async fn tpcds_sf1_host_arrow_join_boundary_queries_match_with_partial_native_children() {
    let root = Path::new(TPCDS_SF1_FIXTURE);
    assert!(root.join("store_sales.parquet").is_file(), "TPC-DS SF1 fixture is missing at {}", root.display());

    let query_filter = vec![
        "q02".to_string(),
        "q10".to_string(),
        "q17".to_string(),
        "q35".to_string(),
        "q45".to_string(),
        "q54".to_string(),
        "q71".to_string(),
        "q75".to_string(),
    ];
    let items = tpcds_parity_workload_items(
        root,
        tpcds_report_session_config(),
        sf1_native_optimizer_config(),
        Some(&query_filter),
    )
    .await
    .unwrap();
    let report = NativeTpcdsParityReport::collect(items, ParityFloatTolerance::exact());
    assert_eq!(report.rows.len(), 8);

    for (query_id, output_rows) in
        [("q02", 2513), ("q10", 6), ("q17", 0), ("q35", 100), ("q45", 19), ("q54", 1), ("q71", 1031), ("q75", 100)]
    {
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
}

#[tokio::test]
async fn tpcds_sf1_host_arrow_window_boundary_queries_match_with_partial_native_children() {
    let root = Path::new(TPCDS_SF1_FIXTURE);
    assert!(root.join("store_sales.parquet").is_file(), "TPC-DS SF1 fixture is missing at {}", root.display());

    let query_filter = vec![
        "q12".to_string(),
        "q20".to_string(),
        "q36".to_string(),
        "q51".to_string(),
        "q53".to_string(),
        "q63".to_string(),
        "q67".to_string(),
        "q70".to_string(),
        "q86".to_string(),
        "q89".to_string(),
        "q98".to_string(),
    ];
    let items = tpcds_parity_workload_items(
        root,
        tpcds_report_session_config(),
        sf1_native_optimizer_config(),
        Some(&query_filter),
    )
    .await
    .unwrap();
    let report = NativeTpcdsParityReport::collect(items, ParityFloatTolerance::exact());
    assert_eq!(report.rows.len(), 11);

    for (query_id, output_rows) in [
        ("q12", 100),
        ("q20", 100),
        ("q36", 100),
        ("q51", 100),
        ("q53", 100),
        ("q63", 100),
        ("q67", 100),
        ("q70", 3),
        ("q86", 100),
        ("q89", 100),
        ("q98", 2521),
    ] {
        let row = report.rows.iter().find(|row| row.query_id == query_id).expect("query row should exist");
        assert_eq!(row.cpu_planning_status, ParityStageStatus::Succeeded, "{query_id}");
        assert_eq!(row.cpu_runtime_status, ParityStageStatus::Succeeded, "{query_id}");
        assert_eq!(row.native_planning_outcome, NativeTpcdsParityPlanningOutcome::AcceptedReplaced, "{query_id}");
        assert_eq!(row.native_reason_code, None, "{query_id}");
        assert_eq!(row.native_unsupported_category, None, "{query_id}");
        assert!(row.native_runtime_started, "{query_id} must execute with partial native children under window");
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
async fn tpcds_sf1_distributed_aggregate_boundary_queries_match_with_partial_native_children() {
    let root = Path::new(TPCDS_SF1_FIXTURE);
    assert!(root.join("store_sales.parquet").is_file(), "TPC-DS SF1 fixture is missing at {}", root.display());

    let query_filter = vec!["q05".to_string(), "q14".to_string(), "q77".to_string(), "q80".to_string()];
    let items = tpcds_parity_workload_items(
        root,
        tpcds_report_session_config(),
        sf1_native_optimizer_config(),
        Some(&query_filter),
    )
    .await
    .unwrap();
    let report = NativeTpcdsParityReport::collect(items, ParityFloatTolerance::exact());
    assert_eq!(report.rows.len(), 4);

    for (query_id, output_rows) in [("q05", 100), ("q14", 100), ("q77", 44), ("q80", 100)] {
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
}

#[tokio::test]
async fn tpcds_sf1_union_boundary_queries_match_with_partial_native_children() {
    let root = Path::new(TPCDS_SF1_FIXTURE);
    assert!(root.join("store_sales.parquet").is_file(), "TPC-DS SF1 fixture is missing at {}", root.display());

    let query_filter = vec![
        "q23".to_string(),
        "q27".to_string(),
        "q33".to_string(),
        "q49".to_string(),
        "q56".to_string(),
        "q60".to_string(),
        "q66".to_string(),
        "q76".to_string(),
    ];
    let items = tpcds_parity_workload_items(
        root,
        tpcds_report_session_config(),
        sf1_native_optimizer_config(),
        Some(&query_filter),
    )
    .await
    .unwrap();
    let report = NativeTpcdsParityReport::collect(items, ParityFloatTolerance::exact());
    assert_eq!(report.rows.len(), 8);

    for (query_id, output_rows) in
        [("q23", 4), ("q27", 100), ("q33", 100), ("q49", 34), ("q56", 100), ("q60", 100), ("q66", 5), ("q76", 100)]
    {
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
}

#[tokio::test]
async fn tpcds_sf1_projection_boundary_queries_with_cpu_scalar_exprs_keep_partial_native_children() {
    let root = Path::new(TPCDS_SF1_FIXTURE);
    assert!(root.join("store_sales.parquet").is_file(), "TPC-DS SF1 fixture is missing at {}", root.display());

    let query_filter = vec!["q78".to_string(), "q84".to_string()];
    let items = tpcds_parity_workload_items(
        root,
        tpcds_report_session_config(),
        sf1_native_optimizer_config(),
        Some(&query_filter),
    )
    .await
    .unwrap();
    let report = NativeTpcdsParityReport::collect(items, ParityFloatTolerance::exact());
    assert_eq!(report.rows.len(), 2);

    for (query_id, output_rows) in [("q78", 100), ("q84", 16)] {
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
}

#[tokio::test]
async fn tpcds_sf1_distinct_aggregate_global_limit_boundaries_keep_partial_native_children() {
    let root = Path::new(TPCDS_SF1_FIXTURE);
    assert!(root.join("store_sales.parquet").is_file(), "TPC-DS SF1 fixture is missing at {}", root.display());

    let query_filter = vec!["q28".to_string()];
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

    let row = &report.rows[0];
    assert_eq!(row.query_id, "q28");
    assert_eq!(row.cpu_planning_status, ParityStageStatus::Succeeded);
    assert_eq!(row.cpu_runtime_status, ParityStageStatus::Succeeded);
    assert_eq!(row.native_planning_outcome, NativeTpcdsParityPlanningOutcome::AcceptedReplaced);
    assert_eq!(row.native_reason_code, None);
    assert_eq!(row.native_unsupported_category, None);
    assert!(row.native_runtime_started, "q28 must execute with partial native children below GlobalLimitExec");
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
async fn tpcds_sf1_strict_rejected_inner_hash_join_boundaries_keep_partial_native_children() {
    let root = Path::new(TPCDS_SF1_FIXTURE);
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
        tpcds_report_session_config(),
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
        assert_eq!(row.cpu_planning_status, ParityStageStatus::Succeeded, "{query_id}");
        assert_eq!(row.cpu_runtime_status, ParityStageStatus::Succeeded, "{query_id}");
        assert_eq!(row.native_planning_outcome, NativeTpcdsParityPlanningOutcome::AcceptedReplaced, "{query_id}");
        assert_eq!(row.native_reason_code, None, "{query_id}");
        assert_eq!(row.native_unsupported_category, None, "{query_id}");
        assert!(row.native_runtime_started, "{query_id} must execute with partial native children");
        assert_eq!(row.native_runtime_outcome, NativeDataFusionRuntimeOutcome::Succeeded, "{query_id}");
        assert_eq!(row.parity_status, ParityStatus::Matched, "{query_id}");
        assert_eq!(row.comparison_mode, comparison_mode, "{query_id}");
        assert_eq!(row.cpu_output_rows, Some(output_rows), "{query_id}");
        assert_eq!(row.native_output_rows, Some(output_rows), "{query_id}");
        assert_eq!(row.native_reservation_denied_count, Some(0), "{query_id}");
        assert_eq!(row.native_retry_count, Some(0), "{query_id}");
        assert_eq!(row.native_runtime_failure, None, "{query_id}");
    }
}

#[tokio::test]
async fn tpcds_sf1_outer_hash_join_boundaries_keep_partial_native_children() {
    let root = Path::new(TPCDS_SF1_FIXTURE);
    assert!(root.join("catalog_sales.parquet").is_file(), "TPC-DS SF1 fixture is missing at {}", root.display());

    let query_filter = vec!["q72".to_string()];
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

    let row = &report.rows[0];
    assert_eq!(row.query_id, "q72");
    assert_eq!(row.cpu_planning_status, ParityStageStatus::Succeeded);
    assert_eq!(row.cpu_runtime_status, ParityStageStatus::Succeeded);
    assert_eq!(row.native_planning_outcome, NativeTpcdsParityPlanningOutcome::AcceptedReplaced);
    assert_eq!(row.native_reason_code, None);
    assert_eq!(row.native_unsupported_category, None);
    assert!(row.native_runtime_started, "q72 must execute with partial native children under CPU outer joins");
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
async fn tpcds_sf1_precision_sensitive_stddev_matches_with_native_order_contract() {
    let root = Path::new(TPCDS_SF1_FIXTURE);
    assert!(root.join("store_sales.parquet").is_file(), "TPC-DS SF1 fixture is missing at {}", root.display());

    let query_filter = vec!["q39".to_string()];
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
    assert_eq!(q39.cpu_planning_status, ParityStageStatus::Succeeded);
    assert_eq!(q39.cpu_runtime_status, ParityStageStatus::Succeeded);
    assert_eq!(q39.native_planning_outcome, NativeTpcdsParityPlanningOutcome::AcceptedReplaced);
    assert_eq!(q39.native_reason_code, None);
    assert_eq!(q39.native_unsupported_category, None);
    assert!(q39.native_runtime_started);
    assert_eq!(q39.native_runtime_outcome, NativeDataFusionRuntimeOutcome::Succeeded);
    assert_eq!(q39.parity_status, ParityStatus::Matched);
    assert_eq!(q39.comparison_mode, ParityComparisonMode::OrderedExact);
    assert_eq!(q39.cpu_output_rows, Some(243));
    assert_eq!(q39.native_output_rows, Some(243));
    assert_eq!(q39.native_reservation_denied_count, Some(0));
    assert_eq!(q39.native_retry_count, Some(0));
    assert_eq!(q39.native_runtime_failure, None);
}

#[tokio::test]
async fn tpcds_sf1_cross_join_count_subplans_match_after_hidden_join_output_projection() {
    let root = Path::new(TPCDS_SF1_FIXTURE);
    assert!(root.join("web_sales.parquet").is_file(), "TPC-DS SF1 fixture is missing at {}", root.display());

    let query_filter = vec!["q88".to_string(), "q90".to_string()];
    let items = tpcds_parity_workload_items(
        root,
        tpcds_report_session_config(),
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

#[test]
fn compare_record_batches_matched_ordered_exact_for_int_and_string() {
    let schema = int_string_schema();
    let cpu = vec![int_string_batch(&schema, &[1, 2, 3], &[Some("a"), Some("b"), None])];
    let native = vec![int_string_batch(&schema, &[1, 2, 3], &[Some("a"), Some("b"), None])];

    let result = compare_record_batches(&schema, &cpu, &schema, &native, true, ParityFloatTolerance::exact());

    assert_eq!(result.status, ParityStatus::Matched);
    assert_eq!(result.mode, ParityComparisonMode::OrderedExact);
    assert!(result.detail.is_none());
    assert_eq!(result.cpu_fingerprint, result.native_fingerprint);
}

#[test]
fn compare_record_batches_mismatched_ordered_exact_first_mismatch_row_and_column() {
    let schema = int_string_schema();
    let cpu = vec![int_string_batch(&schema, &[1, 2, 3], &[Some("a"), Some("b"), Some("c")])];
    let native = vec![int_string_batch(&schema, &[1, 2, 3], &[Some("a"), Some("DIFF"), Some("c")])];

    let result = compare_record_batches(&schema, &cpu, &schema, &native, true, ParityFloatTolerance::exact());

    assert_eq!(result.status, ParityStatus::Mismatched);
    assert_eq!(result.mode, ParityComparisonMode::OrderedExact);
    let detail = result.detail.expect("mismatch should have detail");
    assert!(detail.contains("row 1"), "detail must report first mismatch row: {detail}");
    assert!(detail.contains("label"), "detail must include column name: {detail}");
    assert_ne!(result.cpu_fingerprint, result.native_fingerprint);
}

#[test]
fn compare_record_batches_unordered_match_across_batch_boundaries() {
    let schema = int_string_schema();
    let cpu = vec![
        int_string_batch(&schema, &[3, 1], &[Some("c"), Some("a")]),
        int_string_batch(&schema, &[2], &[Some("b")]),
    ];
    let native = vec![int_string_batch(&schema, &[1, 2, 3], &[Some("a"), Some("b"), Some("c")])];

    let result = compare_record_batches(&schema, &cpu, &schema, &native, false, ParityFloatTolerance::exact());

    assert_eq!(result.status, ParityStatus::Matched);
    assert_eq!(result.mode, ParityComparisonMode::UnorderedExact);
}

#[test]
fn compare_record_batches_distinguishes_null_from_empty_string() {
    let schema = int_string_schema();
    let cpu = vec![int_string_batch(&schema, &[1], &[None])];
    let native = vec![int_string_batch(&schema, &[1], &[Some("")])];

    let result = compare_record_batches(&schema, &cpu, &schema, &native, true, ParityFloatTolerance::exact());

    assert_eq!(result.status, ParityStatus::Mismatched);
    let detail = result.detail.expect("mismatch should have detail");
    assert!(detail.contains("label"), "detail must include column name: {detail}");
}

#[test]
fn compare_record_batches_decimal_scale_difference_is_mismatch() {
    let schema_a = Arc::new(Schema::new(vec![Field::new("amount", DataType::Decimal128(15, 2), false)]));
    let schema_b = Arc::new(Schema::new(vec![Field::new("amount", DataType::Decimal128(15, 4), false)]));
    let batch_a = RecordBatch::try_new(
        Arc::clone(&schema_a),
        vec![Arc::new(Decimal128Array::from(vec![100_i128]).with_precision_and_scale(15, 2).unwrap())],
    )
    .unwrap();
    let batch_b = RecordBatch::try_new(
        Arc::clone(&schema_b),
        vec![Arc::new(Decimal128Array::from(vec![10000_i128]).with_precision_and_scale(15, 4).unwrap())],
    )
    .unwrap();

    let result =
        compare_record_batches(&schema_a, &[batch_a], &schema_b, &[batch_b], true, ParityFloatTolerance::exact());

    assert_eq!(result.status, ParityStatus::Mismatched);
    let detail = result.detail.expect("schema mismatch must have detail");
    assert!(detail.contains("type differs"), "detail must explain schema mismatch: {detail}");
}

#[test]
fn external_oracle_decimal_precision_widening_matches_cpu_value() {
    let cpu_schema = Arc::new(Schema::new(vec![Field::new("amount", DataType::Decimal128(17, 2), false)]));
    let duckdb_schema = Arc::new(Schema::new(vec![Field::new("amount", DataType::Decimal128(38, 2), false)]));
    let cpu_batch = RecordBatch::try_new(
        Arc::clone(&cpu_schema),
        vec![Arc::new(Decimal128Array::from(vec![12345_i128]).with_precision_and_scale(17, 2).unwrap())],
    )
    .unwrap();
    let duckdb_batch = RecordBatch::try_new(
        Arc::clone(&duckdb_schema),
        vec![Arc::new(Decimal128Array::from(vec![12345_i128]).with_precision_and_scale(38, 2).unwrap())],
    )
    .unwrap();
    let item = NativeTpcdsParityWorkloadItem {
        query_id: "qxx".to_string(),
        sql: "SELECT amount FROM t ORDER BY amount".to_string(),
        cpu_outcome: NativeTpcdsParityCpuOutcome::Succeeded { schema: cpu_schema, batches: vec![cpu_batch] },
        native_planning: datafusion_cudf::NativeTpcdsParityNativePlanningOutcome::PlanningFailed(
            datafusion_cudf::NativePlanningFailure {
                stage: "physical_planning".to_string(),
                reason_code: "physical_planning_failed".to_string(),
                message: "not relevant".to_string(),
            },
        ),
        native_runtime: NativeTpcdsParityNativeRuntimeOutcome::NotStarted,
        duckdb_output: NativeTpcdsParityExternalOutput::Succeeded {
            schema: duckdb_schema,
            batches: vec![duckdb_batch],
        },
        external-engine_output: NativeTpcdsParityExternalOutput::NotConfigured,
    };

    let report = NativeTpcdsParityReport::collect([item], ParityFloatTolerance::exact());

    assert_eq!(report.rows[0].duckdb_output.cpu_parity_status, ParityStatus::Matched);
}

#[test]
fn compare_record_batches_float_tolerance_matches_close_values() {
    let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Float64, false)]));
    let cpu =
        RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Float64Array::from(vec![0.30000000000000004_f64]))])
            .unwrap();
    let native = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Float64Array::from(vec![0.3_f64]))]).unwrap();

    let exact = compare_record_batches(
        &schema,
        std::slice::from_ref(&cpu),
        &schema,
        std::slice::from_ref(&native),
        true,
        ParityFloatTolerance::exact(),
    );
    assert_eq!(exact.status, ParityStatus::Mismatched);
    assert_eq!(exact.mode, ParityComparisonMode::OrderedExact);

    let tolerant =
        compare_record_batches(&schema, &[cpu], &schema, &[native], true, ParityFloatTolerance::new(1e-9, 0.0));
    assert_eq!(tolerant.status, ParityStatus::Matched);
    assert_eq!(tolerant.mode, ParityComparisonMode::OrderedFloatTolerant);
}

#[test]
fn compare_record_batches_float_tolerance_rejects_distant_values() {
    let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Float64, false)]));
    let cpu = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Float64Array::from(vec![1.0_f64]))]).unwrap();
    let native = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Float64Array::from(vec![2.0_f64]))]).unwrap();

    let result =
        compare_record_batches(&schema, &[cpu], &schema, &[native], true, ParityFloatTolerance::new(1e-3, 1e-3));
    assert_eq!(result.status, ParityStatus::Mismatched);
    assert_eq!(result.mode, ParityComparisonMode::OrderedFloatTolerant);
}

#[test]
fn compare_record_batches_row_count_mismatch_is_explicit() {
    let schema = int_string_schema();
    let cpu = vec![int_string_batch(&schema, &[1, 2], &[Some("a"), Some("b")])];
    let native = vec![int_string_batch(&schema, &[1], &[Some("a")])];

    let result = compare_record_batches(&schema, &cpu, &schema, &native, true, ParityFloatTolerance::exact());
    assert_eq!(result.status, ParityStatus::Mismatched);
    let detail = result.detail.expect("row count mismatch must have detail");
    assert!(detail.contains("row count"), "detail must explain row count: {detail}");
}

#[test]
fn parity_status_for_cpu_failure_flags_cpu_failed_and_skips_comparison() {
    let item = NativeTpcdsParityWorkloadItem {
        query_id: "q01".to_string(),
        sql: "SELECT 1 ORDER BY 1".to_string(),
        cpu_outcome: NativeTpcdsParityCpuOutcome::PlanningFailed(datafusion_cudf::NativePlanningFailure {
            stage: "sql_to_logical_plan".to_string(),
            reason_code: "sql_planning_failed".to_string(),
            message: "boom".to_string(),
        }),
        native_planning: datafusion_cudf::NativeTpcdsParityNativePlanningOutcome::PlanningFailed(
            datafusion_cudf::NativePlanningFailure {
                stage: "sql_to_logical_plan".to_string(),
                reason_code: "sql_planning_failed".to_string(),
                message: "boom".to_string(),
            },
        ),
        native_runtime: NativeTpcdsParityNativeRuntimeOutcome::NotStarted,
        duckdb_output: NativeTpcdsParityExternalOutput::NotConfigured,
        external-engine_output: NativeTpcdsParityExternalOutput::NotConfigured,
    };
    let report = NativeTpcdsParityReport::collect([item], ParityFloatTolerance::exact());
    assert_eq!(report.rows.len(), 1);
    assert_eq!(report.rows[0].parity_status, ParityStatus::CpuFailed);
    assert_eq!(report.rows[0].comparison_mode, ParityComparisonMode::NotApplicable);
}

#[test]
fn sql_has_top_level_order_by_detects_outer_order_by() {
    assert!(sql_has_top_level_order_by("SELECT * FROM t ORDER BY a"));
    assert!(sql_has_top_level_order_by("\nSELECT * FROM t\nORDER BY a\nLIMIT 100\n"));
}

#[test]
fn sql_has_top_level_order_by_ignores_order_by_inside_subquery() {
    let sql = "SELECT * FROM (SELECT * FROM t ORDER BY a) sub";
    assert!(!sql_has_top_level_order_by(sql));
}

#[test]
fn sql_has_top_level_order_by_ignores_order_by_inside_string_or_comment() {
    let sql = "SELECT 'order by foo' AS x FROM t -- order by trailing comment";
    assert!(!sql_has_top_level_order_by(sql));
    let block = "SELECT /* ORDER BY ignored */ 1 FROM t";
    assert!(!sql_has_top_level_order_by(block));
}

fn int_string_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false), Field::new("label", DataType::Utf8, true)]))
}

fn int_string_batch(schema: &Arc<Schema>, ids: &[i64], labels: &[Option<&str>]) -> RecordBatch {
    RecordBatch::try_new(
        Arc::clone(schema),
        vec![Arc::new(Int64Array::from(ids.to_vec())), Arc::new(StringArray::from(labels.to_vec()))],
    )
    .unwrap()
}

async fn plain_tpcds_physical_plan(root: &Path, query_id: &str) -> Arc<dyn ExecutionPlan> {
    let ctx = SessionContext::new_with_config(tpcds_report_session_config());
    for &table in TPCDS_TABLES {
        let layout = discover_tpcds_table_layout(root, table).unwrap();
        ctx.register_parquet(table, layout.anchor.to_str().unwrap(), ParquetReadOptions::default()).await.unwrap();
    }
    let query = TPCDS_QUERIES.iter().find(|query| query.id == query_id).unwrap();
    let df = ctx.sql(query.sql).await.unwrap();
    df.create_physical_plan().await.unwrap()
}

fn sf1_native_optimizer_config() -> CudfOptimizerConfig {
    CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred)
        .with_native_execution_config(NativeCudfExecutionConfig::bounded(94_489_280_512, 64 * 1024 * 1024, 1, 3, 1))
}
