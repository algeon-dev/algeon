use super::*;

#[tokio::test]
async fn supported_sql_benchmark_plan_reports_datafusion_native_accepted() {
    let dir = unique_benchmark_dir(&std::env::temp_dir());
    let config = test_config();
    let path = benchmark_input_path(&dir, &config);
    write_benchmark_parquet_dataset(&path, &config).unwrap();

    let planned = plan_datafusion_native_sql(&path, &benchmark_sql(&config), &config).await.unwrap();

    assert_eq!(planned.plan.name(), "NativeCudfExec");
    assert!(planned.plan.as_any().downcast_ref::<NativeCudfExec>().is_some());
    assert_eq!(planned.planning_report.accepted_count(), 1);
    assert_eq!(planned.planning_report.rejected_count(), 0);
    let row = &planned.planning_report.rows[0];
    assert_eq!(row.outcome.as_str(), "accepted_replaced");
    assert_eq!(row.post_native_plan_root, "NativeCudfExec");
    assert_eq!(row.post_native_execution_path.as_str(), "native");
    assert!(!row.rejected_before_execution);
    assert!(row.source_diagnostics.iter().any(|diagnostic| diagnostic.direct_device_read));

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn rejected_sql_benchmark_plan_reports_reason_without_native_runtime() {
    let dir = unique_benchmark_dir(&std::env::temp_dir());
    let config = test_config();
    let path = benchmark_input_path(&dir, &config);
    write_benchmark_parquet_dataset(&path, &config).unwrap();

    let planned = plan_datafusion_native_sql(&path, rejected_benchmark_sql(), &config).await.unwrap();

    assert_ne!(planned.plan.name(), "NativeCudfExec");
    assert!(planned.plan.as_any().downcast_ref::<NativeCudfExec>().is_none());
    assert_eq!(planned.planning_report.accepted_count(), 0);
    assert_eq!(planned.planning_report.rejected_count(), 1);
    let row = &planned.planning_report.rows[0];
    assert_eq!(row.outcome.as_str(), "rejected_before_execution");
    assert!(row.reason_code.is_some());
    assert!(row.rejected_before_execution);
    assert_ne!(row.post_native_execution_path.as_str(), "native");
    assert!(planned.plan.metrics().is_none_or(|metrics| metrics.sum_by_name("host_to_device_uploads").is_none()));

    let mut rejected_config = config.clone();
    rejected_config.expected_outcome = BenchmarkExpectedOutcome::PlanningRejected;
    let report = run_expected_planning_rejection_probe(
        &path,
        parquet_dataset_bytes(&path).unwrap(),
        parquet_dataset_layout(&path).unwrap(),
        "files=4".to_string(),
        &rejected_config,
    )
    .await
    .unwrap();
    let text = report.to_text();
    assert!(text.contains("case=datafusion_native_exec_planning_rejected"));
    assert!(text.contains("native_runtime_executed=false"));
    assert!(text.contains("datafusion_native_exec_planning_outcome=rejected_before_execution"));
    assert!(text.contains("datafusion_native_exec_planning_reason_code="));
    assert!(text.contains("datafusion_native_exec_planning_rejected_before_execution=true"));

    fs::remove_dir_all(dir).unwrap();
}
