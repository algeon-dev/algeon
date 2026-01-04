use super::*;

#[tokio::test]
async fn direct_friendly_sql_avg_aggregate_lowers_to_native_avg() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-avg");
    let path = dir.join("input.parquet");
    write_float_measure_parquet_file(&path);
    let plan = sql_plan(&path, direct_friendly_sql_session_config(), SQL_AVG_AGGREGATE_QUERY).await;

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.candidate_count(), 1);
    assert_eq!(result.rewritten_count(), 1);
    assert_eq!(result.rejected_count(), 0);
    let aggregate_exprs = native_aggregate_exprs(&result.plans[0]);
    assert_eq!(aggregate_exprs.len(), 1);
    assert_eq!(aggregate_exprs[0].function, AggregateFunction::Avg);
    assert_eq!(aggregate_exprs[0].alias, "avg(bench.m00)");
    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(source.read_columns, vec!["key".to_string(), "m00".to_string(), "payload".to_string()]);
    assert!(source.diagnostics.direct_device_read);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn direct_friendly_sql_global_avg_lowers_to_native_avg() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-global-avg");
    let path = dir.join("input.parquet");
    write_float_measure_parquet_file(&path);
    let plan = sql_plan(&path, direct_friendly_sql_session_config(), SQL_GLOBAL_AVG_QUERY).await;

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.candidate_count(), 1);
    assert_eq!(result.rewritten_count(), 1);
    assert_eq!(result.rejected_count(), 0);
    assert!(native_aggregate_group_exprs(&result.plans[0]).is_empty());
    let aggregate_exprs = native_aggregate_exprs(&result.plans[0]);
    assert_eq!(aggregate_exprs.len(), 1);
    assert_eq!(aggregate_exprs[0].function, AggregateFunction::Avg);
    assert_eq!(aggregate_exprs[0].alias, "avg(bench.m00)");
    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(source.read_columns, vec!["m00".to_string(), "payload".to_string()]);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn direct_friendly_sql_global_aggregate_parity_functions_lower() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-global-aggregate-parity");
    let path = dir.join("input.parquet");
    write_float_measure_parquet_file(&path);
    let plan = sql_plan(&path, direct_friendly_sql_session_config(), SQL_GLOBAL_AGGREGATE_PARITY_QUERY).await;

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.candidate_count(), 1);
    assert_eq!(result.rewritten_count(), 1);
    assert_eq!(result.rejected_count(), 0);
    assert!(native_aggregate_group_exprs(&result.plans[0]).is_empty());
    assert_eq!(
        native_aggregate_functions(&result.plans[0]),
        vec![
            AggregateFunction::Sum,
            AggregateFunction::Min,
            AggregateFunction::Max,
            AggregateFunction::Count,
            AggregateFunction::Avg,
        ]
    );
    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(source.read_columns, vec!["m00".to_string(), "m01".to_string(), "payload".to_string()]);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn direct_friendly_sql_q6_shaped_global_decimal_sum_lowers() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-q6-shaped");
    let path = dir.join("lineitem.parquet");
    write_q6_lineitem_parquet_file(&path);
    let plan = sql_plan(&path, direct_friendly_sql_session_config(), SQL_Q6_SHAPED_QUERY).await;

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.candidate_count(), 1);
    assert_eq!(result.rewritten_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.rejected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let CandidateOutcome::Rewritten { candidate } = &result.report.outcomes()[0] else {
        panic!("Q6-shaped global aggregate should lower to native");
    };
    assert_eq!(candidate.shape, "ProjectionExec(AggregateExec(FilterExec(LocalParquetScan)))");
    assert_eq!(candidate.operator_count, 4);
    assert!(native_aggregate_group_exprs(&result.plans[0]).is_empty());
    let aggregate_exprs = native_aggregate_exprs(&result.plans[0]);
    assert_eq!(aggregate_exprs.len(), 1);
    assert_eq!(aggregate_exprs[0].function, AggregateFunction::Sum);
    assert!(matches!(
        &aggregate_exprs[0].expr,
        Expr::Column(column) if column.name().starts_with("__cudf_aggregate_arg_")
    ));
    let source = native_local_parquet_source(&result.plans[0]);
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
async fn direct_friendly_sql_filtered_avg_is_rejected() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-filtered-avg");
    let path = dir.join("input.parquet");
    write_float_measure_parquet_file(&path);
    let plan = sql_plan(&path, direct_friendly_sql_session_config(), SQL_FILTERED_AVG_QUERY).await;

    let result = lower_physical_plan_to_native(plan);

    assert_native_rejection(&result, FallbackReason::UnsupportedExpression);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn direct_friendly_sql_distinct_avg_lowers_as_explicit_aggregate_stack() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-distinct-avg");
    let path = dir.join("input.parquet");
    write_float_measure_parquet_file(&path);
    let plan = sql_plan(&path, direct_friendly_sql_session_config(), SQL_DISTINCT_AVG_QUERY).await;

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.rewritten_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.rejected_count(), 0, "outcomes={:?}", result.report.outcomes());
    assert!(
        native_aggregate_nodes(&result.plans[0]).len() >= 2,
        "distinct AVG should be represented by DataFusion's explicit aggregate stack"
    );

    fs::remove_dir_all(dir).unwrap();
}
