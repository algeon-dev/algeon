use algeon_datafusion::native_report::{
    CandidateOutcomeKind, FinalPlanDisposition, NativeDataFusionRuntimeOutcome,
    bench_support::{
        NATIVE_TPCDS_PARITY_TSV_HEADER, NativeTpcdsParityExternalOutputSummary, NativeTpcdsParityReport,
        NativeTpcdsParityRow, ParityComparisonMode, ParityFloatTolerance, ParityStageStatus, ParityStatus,
    },
};
use query_planning::mode::StatisticalAggregateExecutionMode;

#[test]
fn parity_tsv_header_pins_columns() {
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
            "native_retry_count",
            "native_runtime_failure",
        ]
    );
}

#[test]
fn parity_report_to_tsv_emits_metadata_and_header() {
    let row = NativeTpcdsParityRow {
        query_id: "q01".to_string(),
        cpu_planning_status: ParityStageStatus::Succeeded,
        cpu_runtime_status: ParityStageStatus::Succeeded,
        native_planning_outcome: Some(CandidateOutcomeKind::NotSupported),
        native_reason_code: Some("unsupported_operator".to_string()),
        native_unsupported_category: Some("operator_join".to_string()),
        final_disposition: Some(FinalPlanDisposition::DataFusion),
        native_plan_fragment_count: Some(0),
        host_datafusion_boundary_count: Some(0),
        native_runtime_started: false,
        native_runtime_outcome: NativeDataFusionRuntimeOutcome::NotStarted,
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
        native_retry_count: None,
        native_runtime_failure: None,
    };
    let report = NativeTpcdsParityReport::from_rows(vec![row], ParityFloatTolerance::exact())
        .with_statistical_aggregate_execution_mode(StatisticalAggregateExecutionMode::GpuNativeTolerant);
    let tsv = report.to_tsv();
    let first_line = tsv.lines().next().unwrap();
    assert_eq!(first_line, "# parity_float_tolerance_absolute=0");
    assert!(tsv.contains("# statistical_aggregate_execution_mode=gpu_native_tolerant\n"));
    assert!(tsv.contains("# statistical_aggregate_tolerance_contract=float_tolerant\n"));
    assert!(tsv.contains(NATIVE_TPCDS_PARITY_TSV_HEADER));
    // 28 columns separated by tabs in the data row.
    let data_line = tsv.lines().find(|line| line.starts_with("q01\t")).expect("data row should exist");
    assert_eq!(data_line.split('\t').count(), 28);
}
