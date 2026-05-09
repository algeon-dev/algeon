use super::*;

#[tokio::test]
async fn direct_friendly_sql_physical_plan_has_stable_native_lowering_outcome() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-sql-friendly");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let plan = sql_benchmark_plan(&path, direct_friendly_sql_session_config()).await;

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.candidate_count(), 1);
    assert_eq!(result.rewritten_count(), 1);
    assert_eq!(result.rejected_count(), 0);
    let CandidateOutcome::Rewritten { candidate } = &result.report.outcomes()[0] else {
        panic!("direct-friendly SQL plan should lower to native");
    };
    assert_eq!(candidate.shape, "SortExec(fetch)(ProjectionExec(AggregateExec(FilterExec(LocalParquetScan))))");
    assert_eq!(candidate.operator_count, 5);
    assert_eq!(result.plans.len(), 1);
    assert_eq!(result.source_diagnostics.len(), 1);
    assert_eq!(result.source_diagnostics[0].file_count, 1);
    assert!(result.source_diagnostics[0].direct_device_read);
    assert_eq!(result.source_diagnostics[0].predicate_disposition.as_str(), "pruning_only");
    assert!(result.source_diagnostics[0].rejected_features.is_empty());

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn direct_friendly_sql_native_aggregate_parity_functions_lower() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-aggregate-parity");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let plan = sql_plan(&path, direct_friendly_sql_session_config(), SQL_NATIVE_AGGREGATE_PARITY_QUERY).await;

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.candidate_count(), 1);
    assert_eq!(result.rewritten_count(), 1);
    assert_eq!(result.rejected_count(), 0);
    let functions = native_aggregate_functions(&result.plans[0]);
    assert_eq!(functions, vec![AggregateFunction::Min, AggregateFunction::Max, AggregateFunction::Count]);
    assert_eq!(result.source_diagnostics.len(), 1);
    assert!(result.source_diagnostics[0].direct_device_read);
    assert_eq!(result.source_diagnostics[0].predicate_disposition.as_str(), "pruning_only");
    assert!(result.source_diagnostics[0].rejected_features.is_empty());
    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(
        source.read_columns,
        vec!["key".to_string(), "m00".to_string(), "m01".to_string(), "payload".to_string()]
    );

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn direct_friendly_sql_count_star_lowers_with_synthetic_argument_column() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-count-star");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let plan = sql_plan(&path, direct_friendly_sql_session_config(), SQL_COUNT_STAR_QUERY).await;

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.rewritten_count(), 1);
    assert_eq!(result.rejected_count(), 0);
    let aggregate_exprs = native_aggregate_exprs(&result.plans[0]);
    assert_eq!(aggregate_exprs.len(), 2);
    assert_eq!(native_aggregate_functions(&result.plans[0]), vec![AggregateFunction::Sum, AggregateFunction::Count]);
    let count_expr = aggregate_exprs
        .iter()
        .find(|expr| expr.function == AggregateFunction::Count)
        .expect("native aggregate should include COUNT(*)");
    assert!(matches!(
        &count_expr.expr,
        Expr::Column(column) if column.name().starts_with("__cudf_aggregate_arg_")
    ));
    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(
        source.read_columns,
        vec!["key".to_string(), "m00".to_string(), "m01".to_string(), "payload".to_string()]
    );
    assert!(source.diagnostics.direct_device_read);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn direct_friendly_sql_zero_column_count_lowers_to_represented_source() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-zero-column-count");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let plan = count_only_sort_limit_plan(parquet_scan(&path));

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.candidate_count(), 1);
    assert_eq!(result.rewritten_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.rejected_count(), 0, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.source_diagnostics.len(), 1);
    assert!(result.source_diagnostics[0].direct_device_read);
    assert_eq!(result.source_diagnostics[0].read_column_count, 0);
    assert_eq!(result.source_diagnostics[0].projected_output_column_count, 0);
    assert_eq!(result.source_diagnostics[0].pure_filter_column_count, 0);
    assert_eq!(result.source_diagnostics[0].zero_column_count_disposition.as_str(), "represented");
    assert!(result.source_diagnostics[0].rejected_features.is_empty());

    let source = native_local_parquet_source(&result.plans[0]);
    assert!(source.is_zero_column_count_source());
    assert!(source.read_columns.is_empty());
    assert!(source.read_column_file_indices.is_empty());
    assert_eq!(source.diagnostics.zero_column_count_disposition.as_str(), "represented");
    let aggregate_exprs = native_aggregate_exprs(&result.plans[0]);
    assert_eq!(aggregate_exprs.len(), 1);
    assert_eq!(aggregate_exprs[0].function, AggregateFunction::Sum);
    assert!(matches!(&aggregate_exprs[0].expr, Expr::Column(column) if column.name() == ZERO_COLUMN_COUNT_COLUMN));

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn sql_count_only_lowers_to_native_zero_column_count_without_manual_plan_shape() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-sql-count-only");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let plan = sql_plan(&path, count_only_sql_session_config(), SQL_COUNT_ONLY_QUERY).await;

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.candidate_count(), 1);
    assert_eq!(result.rewritten_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.rejected_count(), 0, "outcomes={:?}", result.report.outcomes());
    assert_eq!(
        result.report.outcomes()[0],
        CandidateOutcome::Rewritten {
            candidate: datafusion_cudf::GpuCandidate::with_operator_count(
                "ProjectionExec(AggregateExec(LocalParquetScan))",
                1,
                3,
            ),
        }
    );
    assert_eq!(result.source_diagnostics.len(), 1);
    assert!(result.source_diagnostics[0].direct_device_read);
    assert_eq!(result.source_diagnostics[0].read_column_count, 0);
    assert_eq!(result.source_diagnostics[0].zero_column_count_disposition.as_str(), "represented");

    let source = native_local_parquet_source(&result.plans[0]);
    assert!(source.is_zero_column_count_source());
    assert!(source.read_columns.is_empty());
    let aggregate_exprs = native_aggregate_exprs(&result.plans[0]);
    assert_eq!(aggregate_exprs.len(), 1);
    assert_eq!(aggregate_exprs[0].function, AggregateFunction::Sum);
    assert!(matches!(&aggregate_exprs[0].expr, Expr::Column(column) if column.name() == ZERO_COLUMN_COUNT_COLUMN));
    assert!(matches!(result.plans[0].root_node().kind, PlanNode::Projection(_)));

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn direct_friendly_sql_filter_plan_lowers_when_scan_predicate_is_absent() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-sql-filter-rewrite");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let plan = sql_benchmark_plan(&path, direct_friendly_sql_session_config()).await;
    let plan = sql_plan_without_scan_predicate(plan);

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.candidate_count(), 1);
    assert_eq!(result.rewritten_count(), 1);
    assert_eq!(result.rejected_count(), 0);
    let CandidateOutcome::Rewritten { candidate } = &result.report.outcomes()[0] else {
        panic!("direct-friendly SQL filter plan should lower to native");
    };
    assert_eq!(candidate.shape, "SortExec(fetch)(ProjectionExec(AggregateExec(FilterExec(LocalParquetScan))))");
    assert_eq!(candidate.operator_count, 5);
    assert_eq!(result.plans.len(), 1);
    assert_eq!(result.plans[0].node_count(), 8);
    assert_eq!(result.source_diagnostics.len(), 1);
    assert!(result.source_diagnostics[0].direct_device_read);
    assert_eq!(result.source_diagnostics[0].predicate_disposition.as_str(), "absent");
    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(
        source.read_columns,
        vec!["key".to_string(), "m00".to_string(), "m01".to_string(), "payload".to_string()]
    );

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn direct_friendly_sql_not_filter_lowers_to_native_expr() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-sql-not-filter");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let scan = parquet_scan(&path);
    let scan_schema = scan.schema();
    let payload_lt_10 =
        binary(col("payload", &scan_schema).unwrap(), Operator::Lt, lit(ScalarValue::Int32(Some(10))), &scan_schema)
            .unwrap();
    let predicate = Arc::new(NotExpr::new(payload_lt_10));
    let plan = aggregate_sort_limit_plan_with_filter(scan, predicate, Operator::Plus);

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.candidate_count(), 1);
    assert_eq!(result.rewritten_count(), 1);
    assert_eq!(result.rejected_count(), 0);
    let filter = native_filter_expr(&result.plans[0]);
    assert_eq!(filter.to_string(), "NOT (payload < 10)");
    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(
        source.read_columns,
        vec!["key".to_string(), "m00".to_string(), "m01".to_string(), "payload".to_string()]
    );

    fs::remove_dir_all(dir).unwrap();
}
