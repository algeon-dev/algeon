use super::*;
use crate::native::ParquetReadStrategyConfig;

#[tokio::test]
async fn direct_friendly_sql_avg_aggregate_lowers_to_native_avg() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-avg");
    let path = dir.join("input.parquet");
    write_float_measure_parquet_file(&path);
    let plan = sql_plan(&path, direct_friendly_sql_session_config(), SQL_AVG_AGGREGATE_QUERY).await;

    let result = lower_physical_plan_to_native(plan, &ParquetReadStrategyConfig::default()).unwrap();

    assert_eq!(result.candidate_count(), 1);
    assert_eq!(result.selected_count(), 1);
    assert_eq!(result.not_selected_count(), 0);
    let aggregate_exprs = native_aggregate_exprs(result.selected_plan());
    assert_eq!(aggregate_exprs.len(), 1);
    assert_eq!(aggregate_exprs[0].function, AggregateFunction::Avg);
    assert_eq!(aggregate_exprs[0].alias, "avg(bench.m00)");
    let source = native_local_parquet_source(result.selected_plan());
    assert_eq!(source.read_columns, vec!["key".to_string(), "m00".to_string(), "payload".to_string()]);
    assert!(source.diagnostics.direct_device_read);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn direct_friendly_sql_global_avg_lowers_to_native_avg() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-global-avg");
    let path = dir.join("input.parquet");
    write_float_measure_parquet_file(&path);
    let plan = sql_plan(&path, direct_friendly_sql_session_config(), SQL_GLOBAL_AVG_QUERY).await;

    let result = lower_physical_plan_to_native(plan, &ParquetReadStrategyConfig::default()).unwrap();

    assert_eq!(result.candidate_count(), 1);
    assert_eq!(result.selected_count(), 1);
    assert_eq!(result.not_selected_count(), 0);
    assert!(native_aggregate_group_exprs(result.selected_plan()).is_empty());
    let aggregate_exprs = native_aggregate_exprs(result.selected_plan());
    assert_eq!(aggregate_exprs.len(), 1);
    assert_eq!(aggregate_exprs[0].function, AggregateFunction::Avg);
    assert_eq!(aggregate_exprs[0].alias, "avg(bench.m00)");
    let source = native_local_parquet_source(result.selected_plan());
    assert_eq!(source.read_columns, vec!["m00".to_string(), "payload".to_string()]);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn direct_friendly_sql_global_median_lowers_to_native_median() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-global-median");
    let path = dir.join("input.parquet");
    write_float_measure_parquet_file(&path);
    let plan = sql_plan(&path, direct_friendly_sql_session_config(), SQL_GLOBAL_MEDIAN_QUERY).await;

    let result = lower_physical_plan_to_native(plan, &ParquetReadStrategyConfig::default()).unwrap();

    assert_eq!(result.candidate_count(), 1);
    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    assert!(native_aggregate_group_exprs(result.selected_plan()).is_empty());
    let aggregate_exprs = native_aggregate_exprs(result.selected_plan());
    assert_eq!(aggregate_exprs.len(), 1);
    assert_eq!(aggregate_exprs[0].function, AggregateFunction::Median);
    assert_eq!(aggregate_exprs[0].alias, "median(bench.m00)");
    let source = native_local_parquet_source(result.selected_plan());
    assert_eq!(source.read_columns, vec!["m00".to_string(), "payload".to_string()]);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn direct_friendly_sql_empty_global_mixed_median_and_statistical_aggregate_is_rejected() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-empty-global-mixed-median-statistical");
    let path = dir.join("input.parquet");
    write_float_measure_parquet_file(&path);
    let plan = sql_plan(
        &path,
        direct_friendly_sql_session_config(),
        "SELECT STDDEV_SAMP(m00) AS stdev_m00, MEDIAN(m00) AS median_m00 FROM bench",
    )
    .await;
    let plan = replace_parquet_scans_with_empty_file_groups(plan);

    let result = lower_physical_plan_to_native(plan, &ParquetReadStrategyConfig::default()).unwrap();

    assert_native_rejection(&result, CapabilityReason::UnsupportedOperator);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn direct_friendly_sql_decimal_median_is_rejected_before_cudf_quantile() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-decimal-median");
    let path = dir.join("lineitem.parquet");
    write_discounted_revenue_lineitem_parquet_file(&path);
    let plan = sql_plan(
        &path,
        direct_friendly_sql_session_config(),
        "SELECT MEDIAN(l_extendedprice) AS median_price FROM bench",
    )
    .await;

    let result = lower_physical_plan_to_native(plan, &ParquetReadStrategyConfig::default()).unwrap();

    assert_native_rejection(&result, CapabilityReason::UnsupportedType);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn direct_friendly_sql_global_aggregate_parity_functions_lower() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-global-aggregate-parity");
    let path = dir.join("input.parquet");
    write_float_measure_parquet_file(&path);
    let plan = sql_plan(&path, direct_friendly_sql_session_config(), SQL_GLOBAL_AGGREGATE_PARITY_QUERY).await;

    let result = lower_physical_plan_to_native(plan, &ParquetReadStrategyConfig::default()).unwrap();

    assert_eq!(result.candidate_count(), 1);
    assert_eq!(result.selected_count(), 1);
    assert_eq!(result.not_selected_count(), 0);
    assert!(native_aggregate_group_exprs(result.selected_plan()).is_empty());
    assert_eq!(
        native_aggregate_functions(result.selected_plan()),
        vec![
            AggregateFunction::Sum,
            AggregateFunction::Min,
            AggregateFunction::Max,
            AggregateFunction::Count,
            AggregateFunction::Avg,
        ]
    );
    let source = native_local_parquet_source(result.selected_plan());
    assert_eq!(source.read_columns, vec!["m00".to_string(), "m01".to_string(), "payload".to_string()]);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn direct_friendly_sql_discounted_revenue_global_decimal_sum_lowers() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-discounted-revenue");
    let path = dir.join("lineitem.parquet");
    write_discounted_revenue_lineitem_parquet_file(&path);
    let plan = sql_plan(&path, direct_friendly_sql_session_config(), SQL_Q6_SHAPED_QUERY).await;

    let result = lower_physical_plan_to_native(plan, &ParquetReadStrategyConfig::default()).unwrap();

    assert_eq!(result.candidate_count(), 1);
    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let CandidateOutcome::Selected { candidate } = &result.report.outcomes()[0] else {
        panic!("discounted revenue global aggregate should lower to native");
    };
    assert_eq!(candidate.shape.to_string(), "ProjectionExec(AggregateExec(FilterExec(LocalParquetScan)))");
    assert_eq!(candidate.operator_count(), 4);
    assert!(native_aggregate_group_exprs(result.selected_plan()).is_empty());
    let aggregate_exprs = native_aggregate_exprs(result.selected_plan());
    assert_eq!(aggregate_exprs.len(), 1);
    assert_eq!(aggregate_exprs[0].function, AggregateFunction::Sum);
    assert!(matches!(
        &aggregate_exprs[0].expr,
        Expr::Column(column) if column.name().starts_with("__algeon_native_aggregate_arg_")
    ));
    let source = native_local_parquet_source(result.selected_plan());
    assert_eq!(
        source.read_columns,
        vec![
            "l_discount".to_string(),
            "l_extendedprice".to_string(),
            "l_quantity".to_string(),
            "l_shipdate".to_string(),
        ]
    );
    assert!(source.diagnostics.direct_device_read);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn direct_friendly_sql_filtered_avg_lowers() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-filtered-avg");
    let path = dir.join("input.parquet");
    write_float_measure_parquet_file(&path);
    let plan = sql_plan(&path, direct_friendly_sql_session_config(), SQL_FILTERED_AVG_QUERY).await;

    let result = lower_physical_plan_to_native(plan, &ParquetReadStrategyConfig::default()).unwrap();

    assert_eq!(result.candidate_count(), 1);
    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let aggregates = native_aggregate_nodes(result.selected_plan());
    assert_eq!(aggregates.len(), 1);
    assert_eq!(aggregates[0].spec.aggr_exprs.len(), 1);
    assert!(aggregates[0].spec.aggr_exprs[0].filter.is_some());

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn direct_friendly_sql_distinct_avg_lowers_as_explicit_aggregate_stack() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-distinct-avg");
    let path = dir.join("input.parquet");
    write_float_measure_parquet_file(&path);
    let plan = sql_plan(&path, direct_friendly_sql_session_config(), SQL_DISTINCT_AVG_QUERY).await;

    let result = lower_physical_plan_to_native(plan, &ParquetReadStrategyConfig::default()).unwrap();

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    assert!(
        native_aggregate_nodes(result.selected_plan()).len() >= 2,
        "distinct AVG should be represented by DataFusion's explicit aggregate stack"
    );

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn direct_friendly_sql_global_list_count_lowers() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-global-list-count");
    let path = dir.join("input.parquet");
    write_list_utf8_parquet_file(&path);
    let plan = sql_plan(
        &path,
        direct_friendly_sql_session_config(),
        "SELECT COUNT(tags) AS tag_rows, COUNT(*) AS all_rows FROM bench",
    )
    .await;

    let result = lower_physical_plan_to_native(plan, &ParquetReadStrategyConfig::default()).unwrap();

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    assert!(native_aggregate_group_exprs(result.selected_plan()).is_empty());
    assert_eq!(
        native_aggregate_functions(result.selected_plan()),
        vec![AggregateFunction::Count, AggregateFunction::Count]
    );

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn direct_friendly_sql_distinct_or_grouped_list_count_remains_rejected() {
    for (name, sql) in [
        ("distinct", "SELECT COUNT(DISTINCT tags) AS tag_rows FROM bench"),
        ("grouped", "SELECT key, COUNT(tags) AS tag_rows FROM bench GROUP BY key"),
    ] {
        let dir = unique_temp_dir(&format!("algeon-datafusion-native-lowering-{name}-list-count"));
        let path = dir.join("input.parquet");
        write_list_utf8_parquet_file(&path);
        let plan = sql_plan(&path, direct_friendly_sql_session_config(), sql).await;

        let result = lower_physical_plan_to_native(plan, &ParquetReadStrategyConfig::default()).unwrap();

        assert_native_rejection(&result, CapabilityReason::UnsupportedType);

        fs::remove_dir_all(dir).unwrap();
    }
}
