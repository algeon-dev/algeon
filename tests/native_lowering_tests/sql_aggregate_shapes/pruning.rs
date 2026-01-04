use super::*;

#[tokio::test]
async fn default_sql_pushed_predicate_prunes_native_row_groups_from_metadata() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-sql-row-group-pruning");
    let path = dir.join("input.parquet");
    write_row_group_pruning_parquet_file(&path);
    let total_row_groups = parquet_row_group_count(&path);
    let plan = sql_benchmark_plan(&path, SessionConfig::new()).await;

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(total_row_groups, 4);
    assert_eq!(result.rewritten_count(), 1);
    assert_eq!(result.rejected_count(), 0);
    assert_eq!(result.source_diagnostics.len(), 1);
    assert_eq!(result.source_diagnostics[0].predicate_disposition.as_str(), "pruning_only");
    assert_eq!(result.source_diagnostics[0].selected_row_groups, Some(2));
    assert!(result.source_diagnostics[0].selected_row_groups.unwrap() < total_row_groups);

    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(source.files[0].row_groups.as_deref(), Some(&[2, 3][..]));
    assert_eq!(source.diagnostics.selected_row_groups, Some(2));
    assert_eq!(source.selected_row_group_count(), 2);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn direct_friendly_sql_filter_without_scan_predicate_prunes_native_row_groups_from_metadata() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-filter-row-group-pruning");
    let path = dir.join("input.parquet");
    write_row_group_pruning_parquet_file(&path);
    let total_row_groups = parquet_row_group_count(&path);
    let plan = sql_benchmark_plan(&path, direct_friendly_sql_session_config()).await;
    let plan = sql_plan_without_scan_predicate(plan);

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(total_row_groups, 4);
    assert_eq!(result.rewritten_count(), 1);
    assert_eq!(result.rejected_count(), 0);
    assert_eq!(result.source_diagnostics.len(), 1);
    assert_eq!(result.source_diagnostics[0].predicate_disposition.as_str(), "absent");
    assert_eq!(result.source_diagnostics[0].selected_row_groups, Some(2));
    assert!(result.source_diagnostics[0].selected_row_groups.unwrap() < total_row_groups);

    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(source.files[0].row_groups.as_deref(), Some(&[2, 3][..]));
    assert_eq!(source.diagnostics.selected_row_groups, Some(2));
    assert_eq!(source.selected_row_group_count(), 2);

    fs::remove_dir_all(dir).unwrap();
}
