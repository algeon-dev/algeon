use super::*;

#[test]
fn adapter_validation_rejects_like_over_non_string_input() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-like-type-reject");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let scan = parquet_scan(&path);
    let scan_schema = scan.schema();
    let predicate: Arc<dyn PhysicalExpr> = Arc::new(LikeExpr::new(
        false,
        false,
        col("payload", &scan_schema).unwrap(),
        lit(ScalarValue::Utf8(Some("1%".to_string()))),
    ));
    let plan = aggregate_sort_limit_plan_with_filter(scan, predicate, Operator::Plus);

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.rewritten_count(), 0, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.rejected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert!(matches!(
        result.report.outcomes()[0],
        CandidateOutcome::Rejected { reason: FallbackReason::UnsupportedType, .. }
    ));

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn adapter_validation_rejects_extract_over_date64_until_native_dtype_contract_exists() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-date64-extract-reject");
    let path = dir.join("input.parquet");
    write_date64_parquet_file(&path);
    let plan = sql_plan(&path, direct_friendly_sql_session_config(), SQL_DATE64_EXTRACT_QUERY).await;

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.rewritten_count(), 0, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.rejected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert!(matches!(
        result.report.outcomes()[0],
        CandidateOutcome::Rejected { reason: FallbackReason::UnsupportedType, .. }
    ));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn pushed_scan_projection_reads_only_downstream_required_columns() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-scan-projection-pruning");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let scan = parquet_scan_with_projection_indices(&path, vec![0, 1, 2, 3]);
    let plan = aggregate_sort_limit_plan_without_filter(scan, Operator::Plus);

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.rewritten_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.rejected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(source.read_columns, vec!["key".to_string(), "m00".to_string(), "m01".to_string()]);
    assert!(!source.read_columns.contains(&"payload".to_string()));
    assert_eq!(source.diagnostics.read_column_count, 3);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn scalar_expression_filter_predicate_required_source_columns_are_exact() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-scalar_expression-filter-columns");
    let path = dir.join("input.parquet");
    write_parquet_file_with_tpch_like_columns(&path);
    let plan =
        sql_plan(&path, direct_friendly_sql_session_config(), SQL_SOURCE_COLUMN_FILTER_REQUIRED_COLUMNS_QUERY).await;

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.rewritten_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.rejected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(source.read_columns, vec!["key".to_string(), "label".to_string(), "m00".to_string(), "m01".to_string()]);
    assert_eq!(source.projected_output_columns, vec!["key".to_string(), "m00".to_string(), "m01".to_string()]);
    assert_eq!(source.pure_filter_columns, vec!["label".to_string()]);
    assert_eq!(source.diagnostics.read_column_count, 4);
    assert_eq!(source.diagnostics.projected_output_column_count, 3);
    assert_eq!(source.diagnostics.pure_filter_column_count, 1);
    assert!(!source.read_columns.contains(&"payload".to_string()));
    assert!(!source.read_columns.contains(&"amount".to_string()));

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn scalar_expression_pre_aggregate_projection_required_source_columns_are_exact() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-scalar_expression-pre-projection-columns");
    let path = dir.join("input.parquet");
    write_parquet_file_with_tpch_like_columns(&path);
    let plan =
        sql_plan(&path, direct_friendly_sql_session_config(), SQL_SOURCE_COLUMN_PRE_PROJECTION_REQUIRED_COLUMNS_QUERY)
            .await;

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.rewritten_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.rejected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(source.read_columns, vec!["key".to_string(), "label".to_string(), "m00".to_string(), "m01".to_string()]);
    assert!(source.pure_filter_columns.is_empty());
    for unused in ["payload", "amount", "shipdate", "event_ts"] {
        assert!(!source.read_columns.contains(&unused.to_string()));
    }

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn scalar_expression_aggregate_argument_required_source_columns_are_exact() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-scalar_expression-aggregate-argument-columns");
    let path = dir.join("input.parquet");
    write_parquet_file_with_tpch_like_columns(&path);
    let plan = sql_plan(
        &path,
        direct_friendly_sql_session_config(),
        SQL_SOURCE_COLUMN_AGGREGATE_ARGUMENT_REQUIRED_COLUMNS_QUERY,
    )
    .await;

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.rewritten_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.rejected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(source.read_columns, vec!["key".to_string(), "label".to_string(), "m00".to_string(), "m01".to_string()]);
    let aggregate_exprs = native_aggregate_exprs(&result.plans[0]);
    assert_eq!(aggregate_exprs.len(), 1);
    assert!(matches!(
        &aggregate_exprs[0].expr,
        Expr::Column(column) if column.name().starts_with("__cudf_aggregate_arg_")
    ));
    assert!(!source.read_columns.contains(&"payload".to_string()));

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn scalar_expression_group_expression_required_source_columns_are_exact() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-scalar_expression-group-expression-columns");
    let path = dir.join("input.parquet");
    write_parquet_file_with_tpch_like_columns(&path);
    let plan =
        sql_plan(&path, direct_friendly_sql_session_config(), SQL_SOURCE_COLUMN_GROUP_REQUIRED_COLUMNS_QUERY).await;

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.rewritten_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.rejected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(source.read_columns, vec!["label".to_string(), "m00".to_string(), "shipdate".to_string()]);
    for unused in ["key", "payload", "m01", "amount", "event_ts"] {
        assert!(!source.read_columns.contains(&unused.to_string()));
    }

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn pushed_scan_projection_prunes_payload_for_nested_scalar_expression_scalars() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-scalar_expression-scan-projection-columns");
    let path = dir.join("input.parquet");
    write_parquet_file_with_tpch_like_columns(&path);
    let plan = sql_plan(
        &path,
        direct_friendly_sql_session_config(),
        SQL_SOURCE_COLUMN_PUSHED_PROJECTION_REQUIRED_COLUMNS_QUERY,
    )
    .await;

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.rewritten_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.rejected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(source.read_columns, vec!["label".to_string(), "m00".to_string(), "shipdate".to_string()]);
    assert_eq!(source.read_column_file_indices, vec![4, 2, 6]);
    assert!(!source.read_columns.contains(&"payload".to_string()));
    assert!(!source.read_columns.contains(&"amount".to_string()));

    fs::remove_dir_all(dir).unwrap();
}
