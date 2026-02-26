use std::{
    collections::BTreeSet,
    path::{Path, PathBuf},
    sync::Arc,
};

use arrow::{
    array::{Decimal128Array, Float64Array, Int64Array, StringArray},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use datafusion::{
    physical_plan::{ExecutionPlan, empty::EmptyExec, metrics::MetricsSet},
    prelude::{ParquetReadOptions, SessionConfig, SessionContext},
};
use datafusion_nexus::{
    CUDF_TPCDS_PARQUET_DIR_ENV, CudfOptimizerConfig, GpuCandidate, GpuFallbackPolicy, NATIVE_TPCDS_PARITY_TSV_HEADER,
    NATIVE_TPCDS_PARITY_TSV_SCHEMA_VERSION, NativeAdaptivePipelineDiagnostics, NativeCudfExecutionConfig,
    NativeDataFusionPlanningOutcome, NativeDataFusionPlanningReport, NativeDataFusionPostNativeExecutionPath,
    NativeDataFusionRuntimeOutcome, NativePlanningDiagnostics, NativeStrictBoundedContractDiagnostics,
    NativeTpcdsParityCpuOutcome, NativeTpcdsParityExternalOutput, NativeTpcdsParityExternalOutputSummary,
    NativeTpcdsParityNativePlanningOutcome, NativeTpcdsParityNativeRuntimeOutcome, NativeTpcdsParityPlanningOutcome,
    NativeTpcdsParityReport, NativeTpcdsParityRow, NativeTpcdsParityWorkloadItem, ParityComparisonMode,
    ParityFloatTolerance, ParityStageStatus, ParityStatus, PlanningReport, TPCDS_QUERIES, TPCDS_TABLES,
    TpcdsWorkloadResult, compare_record_batches, default_tpcds_sf1_parquet_root, discover_tpcds_table_layout,
    lower_physical_plan_to_native, sql_has_top_level_order_by, sql_top_level_order_by_covers_output_columns,
    tpcds_datafusion_planning_workload_items_with_optimizer_config,
    tpcds_parity_workload_items_with_session_configs as raw_tpcds_parity_workload_items, tpcds_report_session_config,
};

static TPCDS_NATIVE_PARITY_WORKLOAD_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

fn sf1_fixture_root_or_skip() -> Option<PathBuf> {
    if let Ok(root) = std::env::var(CUDF_TPCDS_PARQUET_DIR_ENV) {
        return Some(PathBuf::from(root));
    }

    let root = PathBuf::from(default_tpcds_sf1_parquet_root());
    let complete = TPCDS_TABLES.iter().all(|table| discover_tpcds_table_layout(&root, table).is_ok());
    if complete {
        Some(root)
    } else {
        eprintln!("skipping external SF1 TPC-DS parity: no complete default SF1 fixture at {}", root.display());
        None
    }
}

#[test]
fn parity_tsv_header_pins_schema_version_and_columns() {
    assert_eq!(NATIVE_TPCDS_PARITY_TSV_SCHEMA_VERSION, 4);
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
            "post_native_execution_path",
            "native_plan_fragment_count",
            "host_datafusion_boundary_count",
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
        cpu_planning_status: datafusion_nexus::ParityStageStatus::Succeeded,
        cpu_runtime_status: datafusion_nexus::ParityStageStatus::Succeeded,
        native_planning_outcome: datafusion_nexus::NativeTpcdsParityPlanningOutcome::RejectedBeforeExecution,
        native_reason_code: Some("unsupported_operator".to_string()),
        native_unsupported_category: Some("operator_join".to_string()),
        post_native_execution_path: NativeDataFusionPostNativeExecutionPath::Cpu,
        native_plan_fragment_count: 0,
        host_datafusion_boundary_count: 0,
        native_runtime_started: false,
        native_runtime_outcome: datafusion_nexus::NativeDataFusionRuntimeOutcome::NotStarted,
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
        native_reservation_live_peak_bytes: None,
        native_reservation_denied_count: None,
        native_retry_count: None,
        native_runtime_failure: None,
    };
    let report = NativeTpcdsParityReport::from_rows(vec![row], ParityFloatTolerance::exact())
        .with_statistical_aggregate_execution_mode(
            datafusion_nexus::StatisticalAggregateExecutionMode::GpuNativeTolerant,
        );
    let tsv = report.to_tsv();
    let first_line = tsv.lines().next().unwrap();
    assert_eq!(
        first_line,
        format!("# native_tpcds_parity_tsv_schema_version={NATIVE_TPCDS_PARITY_TSV_SCHEMA_VERSION}")
    );
    assert!(tsv.contains("# statistical_aggregate_execution_mode=gpu_native_tolerant\n"));
    assert!(tsv.contains("# statistical_aggregate_tolerance_contract=float_tolerant\n"));
    assert!(tsv.contains("# stddev_samp_execution_mode=gpu_native_tolerant\n"));
    assert!(tsv.contains(NATIVE_TPCDS_PARITY_TSV_HEADER));
    // 30 columns separated by tabs in the data row.
    let data_line = tsv.lines().find(|line| line.starts_with("q01\t")).expect("data row should exist");
    assert_eq!(data_line.split('\t').count(), 30);
}

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
    let expected: BTreeSet<&str> = BTreeSet::new();

    assert_eq!(partial_native, expected);
    assert_eq!(cpu_only, expected);
    for query_id in ["q34", "q73"] {
        let row =
            report.rows.iter().find(|row| row.query_id == query_id).unwrap_or_else(|| panic!("{query_id} is missing"));
        assert_eq!(row.outcome, NativeDataFusionPlanningOutcome::AcceptedReplaced, "{query_id}");
        assert_eq!(row.post_native_plan_root, "NativeCudfExec", "{query_id}");
        assert_eq!(row.post_native_execution_path, NativeDataFusionPostNativeExecutionPath::Native, "{query_id}");
        assert_eq!(row.native_plan_fragment_count, 1, "{query_id}");
    }
    for query_id in expected {
        let rows: Vec<_> = report.rows.iter().filter(|row| row.query_id == query_id).collect();
        assert!(!rows.is_empty(), "{query_id} should have planning rows");
        assert!(rows.iter().all(|row| row.native_plan_fragment_count > 0), "{query_id} should report native fragments");
        assert!(
            rows.iter().all(|row| row.host_datafusion_boundary_count > 0),
            "{query_id} should report host DataFusion boundary operators"
        );
    }
}

#[tokio::test]
async fn tpcds_sf1_q44_keyless_outer_cross_join_matches_as_whole_native() {
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
    assert_eq!(row.query_id, "q44");
    assert_eq!(row.cpu_planning_status, ParityStageStatus::Succeeded);
    assert_eq!(row.cpu_runtime_status, ParityStageStatus::Succeeded);
    assert_eq!(row.native_planning_outcome, NativeTpcdsParityPlanningOutcome::AcceptedReplaced);
    assert_eq!(row.native_reason_code, None);
    assert_eq!(row.native_unsupported_category, None);
    assert_eq!(row.post_native_execution_path, NativeDataFusionPostNativeExecutionPath::Native);
    assert_eq!(row.native_plan_fragment_count, 1);
    assert_eq!(row.host_datafusion_boundary_count, 0);
    assert!(row.native_runtime_started, "q44 must execute through the native path");
    assert_eq!(row.native_runtime_outcome, NativeDataFusionRuntimeOutcome::Succeeded);
    assert_eq!(row.parity_status, ParityStatus::Matched);
    assert_eq!(row.comparison_mode, ParityComparisonMode::OrderedExact);
    assert_eq!(row.cpu_output_rows, Some(10));
    assert_eq!(row.native_output_rows, Some(10));
    assert_eq!(row.native_reservation_denied_count, Some(0));
    assert_eq!(row.native_retry_count, Some(0));
    assert_eq!(row.native_runtime_failure, None);
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

    for (query_id, output_rows) in [("q70", 3), ("q86", 100)] {
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
async fn tpcds_sf1_rank_window_queries_match_as_whole_native() {
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

    for (query_id, output_rows) in [("q36", 100), ("q47", 100), ("q57", 100), ("q67", 100)] {
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
async fn tpcds_sf1_cumulative_aggregate_window_queries_match_as_whole_native() {
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
async fn tpcds_sf1_projection_scalar_queries_match_as_whole_native() {
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

    for (query_id, output_rows) in [("q02", 2513), ("q78", 100), ("q84", 16)] {
        let row = report.rows.iter().find(|row| row.query_id == query_id).expect("query row should exist");
        assert_eq!(row.cpu_planning_status, ParityStageStatus::Succeeded, "{query_id}");
        assert_eq!(row.cpu_runtime_status, ParityStageStatus::Succeeded, "{query_id}");
        assert_eq!(row.native_planning_outcome, NativeTpcdsParityPlanningOutcome::AcceptedReplaced, "{query_id}");
        assert_eq!(row.native_reason_code, None, "{query_id}");
        assert_eq!(row.native_unsupported_category, None, "{query_id}");
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
async fn tpcds_sf1_distinct_aggregate_global_limit_query_matches_as_whole_native() {
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
    assert_eq!(row.query_id, "q28");
    assert_eq!(row.cpu_planning_status, ParityStageStatus::Succeeded);
    assert_eq!(row.cpu_runtime_status, ParityStageStatus::Succeeded);
    assert_eq!(row.native_planning_outcome, NativeTpcdsParityPlanningOutcome::AcceptedReplaced);
    assert_eq!(row.native_reason_code, None);
    assert_eq!(row.native_unsupported_category, None);
    assert_eq!(row.post_native_execution_path, NativeDataFusionPostNativeExecutionPath::Native);
    assert_eq!(row.native_plan_fragment_count, 1);
    assert_eq!(row.host_datafusion_boundary_count, 0);
    assert!(row.native_runtime_started, "q28 must execute through the native path");
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
async fn tpcds_sf1_hash_join_queries_match_as_whole_native() {
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
        assert_eq!(row.comparison_mode, comparison_mode, "{query_id}");
        assert_eq!(row.cpu_output_rows, Some(output_rows), "{query_id}");
        assert_eq!(row.native_output_rows, Some(output_rows), "{query_id}");
        assert_eq!(row.native_reservation_denied_count, Some(0), "{query_id}");
        assert_eq!(row.native_retry_count, Some(0), "{query_id}");
        assert_eq!(row.native_runtime_failure, None, "{query_id}");
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
async fn tpcds_sf1_precision_sensitive_stddev_matches_with_native_order_contract() {
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
    assert_eq!(result.cpu_fingerprint, result.native_fingerprint);
}

#[test]
fn compare_record_batches_matched_uses_comparison_fingerprints_for_compatible_schema_names() {
    let cpu_schema = int_string_schema();
    let native_schema = Arc::new(Schema::new(vec![
        Field::new("native_id", DataType::Int64, false),
        Field::new("native_label", DataType::Utf8, true),
    ]));
    let cpu = vec![int_string_batch(&cpu_schema, &[1, 2], &[Some("a"), Some("b")])];
    let native = vec![int_string_batch(&native_schema, &[1, 2], &[Some("a"), Some("b")])];

    let result =
        compare_record_batches(&cpu_schema, &cpu, &native_schema, &native, true, ParityFloatTolerance::exact());

    assert_eq!(result.status, ParityStatus::Matched);
    assert_eq!(result.cpu_fingerprint, result.native_fingerprint);
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
        native_planning: datafusion_nexus::NativeTpcdsParityNativePlanningOutcome::PlanningFailed(
            datafusion_nexus::NativePlanningFailure {
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
fn parity_report_retries_unordered_only_for_partial_order_by_ties() {
    let schema = int_string_schema();
    let cpu_batch = int_string_batch(&schema, &[1, 1], &[Some("a"), Some("b")]);
    let native_batch = int_string_batch(&schema, &[1, 1], &[Some("b"), Some("a")]);
    let partial_item = NativeTpcdsParityWorkloadItem {
        query_id: "partial".to_string(),
        sql: "SELECT id, label FROM t ORDER BY id".to_string(),
        cpu_outcome: NativeTpcdsParityCpuOutcome::Succeeded {
            schema: Arc::clone(&schema),
            batches: vec![cpu_batch.clone()],
        },
        native_planning: accepted_native_planning(&schema),
        native_runtime: NativeTpcdsParityNativeRuntimeOutcome::Succeeded {
            schema: Arc::clone(&schema),
            batches: vec![native_batch.clone()],
            metrics: MetricsSet::new(),
        },
        duckdb_output: NativeTpcdsParityExternalOutput::NotConfigured,
    };
    let total_item = NativeTpcdsParityWorkloadItem {
        query_id: "total".to_string(),
        sql: "SELECT id, label FROM t ORDER BY id, label".to_string(),
        cpu_outcome: NativeTpcdsParityCpuOutcome::Succeeded { schema: Arc::clone(&schema), batches: vec![cpu_batch] },
        native_planning: accepted_native_planning(&schema),
        native_runtime: NativeTpcdsParityNativeRuntimeOutcome::Succeeded {
            schema: Arc::clone(&schema),
            batches: vec![native_batch],
            metrics: MetricsSet::new(),
        },
        duckdb_output: NativeTpcdsParityExternalOutput::NotConfigured,
    };

    let report = NativeTpcdsParityReport::collect([partial_item, total_item], ParityFloatTolerance::exact());
    let partial = report.rows.iter().find(|row| row.query_id == "partial").unwrap();
    assert_eq!(partial.parity_status, ParityStatus::Matched);
    assert_eq!(partial.comparison_mode, ParityComparisonMode::UnorderedExact);

    let total = report.rows.iter().find(|row| row.query_id == "total").unwrap();
    assert_eq!(total.parity_status, ParityStatus::Mismatched);
    assert_eq!(total.comparison_mode, ParityComparisonMode::OrderedExact);
}

#[test]
fn parity_status_for_cpu_failure_flags_cpu_failed_and_skips_comparison() {
    let item = NativeTpcdsParityWorkloadItem {
        query_id: "q01".to_string(),
        sql: "SELECT 1 ORDER BY 1".to_string(),
        cpu_outcome: NativeTpcdsParityCpuOutcome::PlanningFailed(datafusion_nexus::NativePlanningFailure {
            stage: "sql_to_logical_plan".to_string(),
            reason_code: "sql_planning_failed".to_string(),
            message: "boom".to_string(),
        }),
        native_planning: datafusion_nexus::NativeTpcdsParityNativePlanningOutcome::PlanningFailed(
            datafusion_nexus::NativePlanningFailure {
                stage: "sql_to_logical_plan".to_string(),
                reason_code: "sql_planning_failed".to_string(),
                message: "boom".to_string(),
            },
        ),
        native_runtime: NativeTpcdsParityNativeRuntimeOutcome::NotStarted,
        duckdb_output: NativeTpcdsParityExternalOutput::NotConfigured,
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

#[test]
fn sql_top_level_order_by_covers_output_columns_detects_total_order() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("label", DataType::Utf8, true),
    ]));
    assert!(sql_top_level_order_by_covers_output_columns("SELECT id, label FROM t ORDER BY id, label", &schema));
    assert!(sql_top_level_order_by_covers_output_columns(
        "SELECT id, label FROM t ORDER BY t.id DESC NULLS FIRST, label ASC LIMIT 10",
        &schema,
    ));
}

#[test]
fn sql_top_level_order_by_covers_output_columns_rejects_partial_order() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("brand_id", DataType::Int64, false),
        Field::new("brand", DataType::Utf8, true),
        Field::new("t_hour", DataType::Int64, false),
        Field::new("t_minute", DataType::Int64, false),
        Field::new("ext_price", DataType::Decimal128(15, 2), true),
    ]));
    let q71_like = "
SELECT i_brand_id brand_id, i_brand brand, t_hour, t_minute, sum(ext_price) ext_price
FROM t
GROUP BY i_brand, i_brand_id, t_hour, t_minute
ORDER BY ext_price DESC NULLS FIRST, i_brand_id NULLS FIRST, t_hour NULLS FIRST
";
    assert!(!sql_top_level_order_by_covers_output_columns(q71_like, &schema));
    assert!(!sql_top_level_order_by_covers_output_columns(
        "SELECT id, label, value FROM t ORDER BY id, label",
        &Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("label", DataType::Utf8, true),
            Field::new("value", DataType::Int64, false),
        ]),
    ));
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

fn accepted_native_planning(schema: &Arc<Schema>) -> NativeTpcdsParityNativePlanningOutcome {
    let mut report = PlanningReport::new();
    report.push_rewritten(GpuCandidate::new("test_native_plan", schema.fields().len()));
    NativeTpcdsParityNativePlanningOutcome::Planned {
        accepted: true,
        planned_plan: Arc::new(EmptyExec::new(Arc::clone(schema))) as Arc<dyn ExecutionPlan>,
        diagnostics: Box::new(NativePlanningDiagnostics {
            input_plan_root: "EmptyExec".to_string(),
            input_plan_shape: "EmptyExec".to_string(),
            report,
            native_optimizer: nexus_query_engine::NativeOptimizerDiagnostics::default(),
            source_diagnostics: Vec::new(),
            strict_bounded_contract: NativeStrictBoundedContractDiagnostics::default(),
            adaptive_pipeline: NativeAdaptivePipelineDiagnostics::default(),
        }),
    }
}

async fn tpcds_parity_workload_items(
    root: &Path,
    cpu_session_config: SessionConfig,
    optimizer_config: CudfOptimizerConfig,
    query_filter: Option<&[String]>,
) -> TpcdsWorkloadResult<Vec<NativeTpcdsParityWorkloadItem>> {
    // The parity workload executes native CUDA/libcudf plans. Those bindings use
    // process-global CUDA/RMM state, so parallel libtest workers in this one test
    // binary can crash the process even when each individual query is valid.
    let _guard = TPCDS_NATIVE_PARITY_WORKLOAD_LOCK.lock().await;
    raw_tpcds_parity_workload_items(
        root,
        cpu_session_config,
        tpcds_report_session_config(),
        optimizer_config,
        query_filter,
    )
    .await
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

fn sf1_tpcds_test_session_config() -> SessionConfig {
    tpcds_report_session_config().with_target_partitions(tpcds_test_target_partitions())
}

// This controls only the CPU DataFusion oracle. The native side keeps the
// report session's single DataFusion partition; GPU chunking is owned by the
// native/libcudf execution config below.
fn tpcds_test_target_partitions() -> usize {
    test_threads_from_args()
        .or_else(test_threads_from_env)
        .unwrap_or_else(|| std::thread::available_parallelism().map(|count| count.get()).unwrap_or(1))
}

fn test_threads_from_args() -> Option<usize> {
    let mut args = std::env::args();
    while let Some(arg) = args.next() {
        if let Some(value) = arg.strip_prefix("--test-threads=") {
            return parse_positive_usize(value);
        }
        if arg == "--test-threads" {
            return args.next().and_then(|value| parse_positive_usize(&value));
        }
    }
    None
}

fn test_threads_from_env() -> Option<usize> {
    std::env::var("RUST_TEST_THREADS").ok().and_then(|value| parse_positive_usize(&value))
}

fn parse_positive_usize(value: &str) -> Option<usize> {
    value.parse::<usize>().ok().filter(|value| *value > 0)
}

fn sf1_native_optimizer_config() -> CudfOptimizerConfig {
    CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred).with_native_execution_config(
        NativeCudfExecutionConfig::bounded(
            NativeCudfExecutionConfig::auto_sized_device_budget_bytes(),
            4 * 1024 * 1024 * 1024,
            1024,
            3,
            1,
        ),
    )
}

fn q72_native_optimizer_config() -> CudfOptimizerConfig {
    CudfOptimizerConfig::new(GpuFallbackPolicy::GpuPreferred).with_native_execution_config(
        NativeCudfExecutionConfig::bounded(
            NativeCudfExecutionConfig::auto_sized_device_budget_bytes(),
            4 * 1024 * 1024 * 1024,
            1024,
            3,
            1,
        ),
    )
}

fn plan_contains_aggregate_function(
    plan: &nexus_query_engine::QueryPlan,
    function: nexus_query_engine::AggregateFunction,
) -> bool {
    plan.nodes().iter().any(|node| match &node.kind {
        nexus_query_engine::PlanNode::Aggregate(aggregate) => {
            aggregate.spec.aggr_exprs.iter().any(|expr| expr.function == function)
        },
        _ => false,
    })
}
