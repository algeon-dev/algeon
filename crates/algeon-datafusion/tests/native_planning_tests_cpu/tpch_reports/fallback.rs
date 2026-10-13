use super::{super::*, tpch_expectations::*};

#[tokio::test]
async fn tpch_fallback_pressure_report_emits_one_pre_execution_reject_per_query() {
    let dir = unique_temp_dir("algeon-datafusion-tpch-fallback-pressure");
    write_tiny_tpch_parquet_dataset(&dir).unwrap();

    let metadata = tiny_tpch_dataset_metadata(&dir).unwrap();
    let items = tpch_fallback_pressure_workload_items(&dir, tpch_report_session_config()).await.unwrap();
    let report = NativeFallbackPressureReport::collect(items)
        .with_workload_dataset_metadata(metadata.to_workload_dataset_metadata());

    assert_eq!(report.rows.len(), TPCH_QUERY_COUNT);
    assert_eq!(report.selected_count(), TPCH_QUERY_COUNT);
    assert_eq!(report.not_selected_count(), 0);

    let actual_ids = report.rows.iter().map(|row| row.query_id.as_str()).collect::<Vec<_>>();
    let expected_ids = tpch_query_ids().collect::<Vec<_>>();
    assert_eq!(actual_ids, expected_ids);

    for row in &report.rows {
        assert_eq!(row.outcome, Some(CandidateOutcomeKind::Selected), "unexpected outcome for {}", row.query_id);
        assert_eq!(row.reason_code.as_deref(), None, "unexpected reason for {}", row.query_id);
        assert_eq!(row.unsupported_category.as_deref(), None, "unexpected unsupported category for {}", row.query_id);
        assert_eq!(row.physical_root, expected_tpch_physical_root(row.query_id.as_str()));
        assert_eq!(
            row.native_candidate_shape.as_deref().unwrap_or("none"),
            expected_tpch_native_candidate_shape(row.query_id.as_str()),
            "unexpected native candidate shape for {}",
            row.query_id
        );
        assert_eq!(
            row.native_candidate_shape_family.as_deref().unwrap_or("none"),
            expected_tpch_native_candidate_shape_family(row.query_id.as_str()),
            "unexpected native candidate shape family for {}",
            row.query_id
        );
        assert!(
            row.physical_shape.contains(expected_tpch_physical_shape_marker(row.query_id.as_str())),
            "unexpected physical shape for {}: {}",
            row.query_id,
            row.physical_shape
        );
        assert_tpch_join_diagnostics(row.query_id.as_str(), &row.join_diagnostics);
        assert!(!row.rejected_before_execution, "unexpected pre-execution rejection flag for {}", row.query_id);
    }
    let q06 = report_row(&report, "q06");
    assert_eq!(
        q06.native_candidate_shape.as_deref(),
        Some("ProjectionExec(AggregateExec(FilterExec(LocalParquetScan)))")
    );
    assert_eq!(q06.native_candidate_shape_family.as_deref(), Some("projected_global_local_parquet_aggregate"));

    let text = report.to_tsv();
    assert!(text.starts_with("# workload=tpch\n"));
    assert!(text.contains("# workload=tpch\n"));
    assert!(text.contains("# dataset_scale_factor=tiny\n"));
    assert!(text.contains("# dataset_provenance=generated_tiny\n"));
    assert!(text.contains("# dataset_generation_command=algeon-datafusion::write_tiny_tpch_parquet_dataset\n"));
    assert!(text.contains("# dataset_table_count=8\n"));
    assert!(text.contains("# dataset_file_count=8\n"));
    assert!(text.contains("# dataset_total_rows=26\n"));
    assert!(text.contains(&format!("# dataset_total_bytes={}\n", metadata.total_bytes)));
    assert!(text.contains(&format!("# dataset_metadata_fingerprint={}\n", metadata.metadata_fingerprint)));
    assert!(text.contains("# dataset_content_fingerprint=not_computed\n"));
    assert!(text.contains(
        "# dataset_table_file_counts=customer=1,lineitem=1,nation=1,orders=1,part=1,partsupp=1,region=1,supplier=1\n"
    ));
    assert!(text.contains(
        "# dataset_table_row_counts=customer=3,lineitem=3,nation=5,orders=3,part=3,partsupp=3,region=3,supplier=3\n"
    ));
    assert!(text.contains(&format!("# dataset_table_byte_sizes={}\n", metadata.table_byte_sizes())));
    assert!(text.contains(&format!("{NATIVE_FALLBACK_PRESSURE_TSV_HEADER}\n")));
    assert!(text.contains("q01\t"));
    assert!(text.contains("q06\t"));
    assert!(text.contains("q22\t"));
    assert!(text.contains("\tselected\t"));

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn tpch_scalar_fallback_pressure_report_pins_expression_categories() {
    let dir = unique_temp_dir("algeon-datafusion-tpch-scalar-fallback-pressure");
    write_tiny_tpch_parquet_dataset(&dir).unwrap();

    let metadata = tiny_tpch_dataset_metadata(&dir).unwrap();
    let items = tpch_scalar_fallback_pressure_workload_items(&dir, tpch_report_session_config()).await.unwrap();
    let report = NativeFallbackPressureReport::collect(items)
        .with_workload_dataset_metadata(metadata.to_workload_dataset_metadata());

    let expected = [
        ("q12_scalar_in_list_null", "expression_in_list"),
        ("q22_scalar_substring_dynamic_start", "expression_substring_argument"),
        ("q07_scalar_extract_quarter", "expression_extract_field"),
        ("q01_scalar_interval_minus_date", "expression_interval_arithmetic"),
    ];
    assert_eq!(report.rows.len(), expected.len());
    assert_eq!(report.selected_count(), 0);
    assert_eq!(report.not_selected_count(), expected.len());

    for (query_id, expected_category) in expected {
        let row = report_row(&report, query_id);
        assert_eq!(row.outcome, Some(CandidateOutcomeKind::NotSupported), "unexpected outcome for {query_id}");
        assert_eq!(row.reason_code.as_deref(), Some("unsupported_expression"), "unexpected reason for {query_id}");
        assert_eq!(row.unsupported_category.as_deref(), Some(expected_category), "unexpected category for {query_id}");
        assert!(!row.rejected_before_execution, "{query_id} should remain on the executable DataFusion baseline");
    }

    let text = report.to_tsv();
    assert!(text.contains("\tunsupported_expression\texpression_in_list\t"));
    assert!(text.contains("\tunsupported_expression\texpression_substring_argument\t"));
    assert!(text.contains("\tunsupported_expression\texpression_extract_field\t"));
    assert!(text.contains("\tunsupported_expression\texpression_interval_arithmetic\t"));

    fs::remove_dir_all(dir).unwrap();
}
