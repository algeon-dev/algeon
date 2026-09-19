use super::*;

#[tokio::test]
async fn default_sql_pushed_predicate_prunes_native_row_groups_from_metadata() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-sql-row-group-pruning");
    let path = dir.join("input.parquet");
    write_row_group_pruning_parquet_file(&path);
    let total_row_groups = parquet_row_group_count(&path);
    let plan = sql_benchmark_plan(&path, SessionConfig::new()).await;

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(total_row_groups, 4);
    assert_eq!(result.selected_count(), 1);
    assert_eq!(result.not_selected_count(), 0);
    assert_eq!(result.source_diagnostics.len(), 1);
    assert_eq!(result.source_diagnostics[0].predicate_disposition.as_str(), "pruning_only");
    assert_eq!(result.source_diagnostics[0].selected_row_groups, Some(2));
    assert!(result.source_diagnostics[0].selected_row_groups.unwrap() < total_row_groups);

    let source = native_local_parquet_source(&result.plans[0]);
    assert!(source.reader_predicate.is_none());
    assert_eq!(source.files[0].row_groups.as_deref(), Some(&[2, 3][..]));
    assert_eq!(source.diagnostics.selected_row_groups, Some(2));
    assert_eq!(source.selected_row_group_count(), 2);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn direct_friendly_sql_filter_without_scan_predicate_prunes_native_row_groups_from_metadata() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-filter-row-group-pruning");
    let path = dir.join("input.parquet");
    write_row_group_pruning_parquet_file(&path);
    let total_row_groups = parquet_row_group_count(&path);
    let plan = sql_benchmark_plan(&path, direct_friendly_sql_session_config()).await;
    let plan = sql_plan_without_scan_predicate(plan);

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(total_row_groups, 4);
    assert_eq!(result.selected_count(), 1);
    assert_eq!(result.not_selected_count(), 0);
    assert_eq!(result.source_diagnostics.len(), 1);
    assert_eq!(result.source_diagnostics[0].predicate_disposition.as_str(), "native_post_scan_only");
    assert_eq!(result.source_diagnostics[0].selected_row_groups, Some(2));
    assert!(result.source_diagnostics[0].selected_row_groups.unwrap() < total_row_groups);

    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(source.files[0].row_groups.as_deref(), Some(&[2, 3][..]));
    assert_eq!(source.diagnostics.selected_row_groups, Some(2));
    assert_eq!(source.selected_row_group_count(), 2);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn direct_friendly_sql_date_conjunction_prunes_native_row_groups_from_metadata() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-date-row-group-pruning");
    let path = dir.join("lineitem.parquet");
    write_discounted_revenue_lineitem_parquet_file(&path);
    let total_row_groups = parquet_row_group_count(&path);
    let plan = sql_plan(&path, direct_friendly_sql_session_config(), SQL_Q6_SHAPED_QUERY).await;

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(total_row_groups, 4);
    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.source_diagnostics[0].predicate_disposition.as_str(), "pruning_only");
    assert_eq!(result.source_diagnostics[0].selected_row_groups, Some(3));
    let source = native_local_parquet_source(&result.plans[0]);
    assert!(source.reader_predicate.is_none());
    assert_eq!(source.files[0].row_groups.as_deref(), Some(&[0, 1, 2][..]));
    assert_eq!(source.selected_row_group_count(), 3);
    assert_eq!(native_filter_exprs(&result.plans[0]).len(), 1);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn direct_friendly_sql_widening_cast_prunes_native_row_groups_from_metadata() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-cast-row-group-pruning");
    let path = dir.join("input.parquet");
    write_row_group_pruning_parquet_file(&path);
    let plan = sql_plan(
        &path,
        direct_friendly_sql_session_config(),
        "
        SELECT key, SUM(m00) AS sum_m00
        FROM bench
        WHERE CAST(payload AS BIGINT) >= 10
        GROUP BY key
        ORDER BY key ASC
        LIMIT 10
        ",
    )
    .await;

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.source_diagnostics[0].selected_row_groups, Some(2));
    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(source.files[0].row_groups.as_deref(), Some(&[2, 3][..]));

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn direct_friendly_sql_in_list_prunes_native_row_groups_from_metadata() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-in-list-row-group-pruning");
    let path = dir.join("input.parquet");
    write_nullable_row_group_pruning_parquet_file(&path);
    let plan = sql_plan(
        &path,
        direct_friendly_sql_session_config(),
        "
        SELECT orderkey, SUM(m00) AS sum_m00
        FROM bench
        WHERE orderkey IN (1, 2)
        GROUP BY orderkey
        ORDER BY orderkey ASC
        LIMIT 10
        ",
    )
    .await;

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.source_diagnostics[0].selected_row_groups, Some(1));
    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(source.files[0].row_groups.as_deref(), Some(&[0][..]));

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn direct_friendly_sql_is_null_prunes_native_row_groups_from_metadata() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-is-null-row-group-pruning");
    let path = dir.join("input.parquet");
    write_nullable_row_group_pruning_parquet_file(&path);
    let plan = sql_plan(
        &path,
        direct_friendly_sql_session_config(),
        "
        SELECT COUNT(*) AS row_count
        FROM bench
        WHERE orderkey IS NULL
        ",
    )
    .await;

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.source_diagnostics[0].selected_row_groups, Some(2));
    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(source.files[0].row_groups.as_deref(), Some(&[1, 2][..]));

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn direct_friendly_sql_nested_prefix_pruning_uses_parquet_leaf_stats() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-nested-row-group-pruning");
    let path = dir.join("input.parquet");
    write_nested_prefix_row_group_pruning_parquet_file(&path);
    let total_row_groups = parquet_row_group_count(&path);
    let plan = sql_plan(
        &path,
        direct_friendly_sql_session_config(),
        "
        SELECT key, SUM(m00) AS sum_m00
        FROM bench
        WHERE key = 200
        GROUP BY key
        ORDER BY key ASC
        LIMIT 10
        ",
    )
    .await;

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(total_row_groups, 2);
    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.source_diagnostics[0].selected_row_groups, Some(1));
    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(source.files[0].row_groups.as_deref(), Some(&[1][..]));

    fs::remove_dir_all(dir).unwrap();
}
