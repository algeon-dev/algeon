use super::*;

#[tokio::test]
async fn direct_friendly_sql_physical_plan_has_stable_native_lowering_outcome() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-sql-friendly");
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
    let source = native_local_parquet_source(&result.plans[0]);
    assert!(source.reader_predicate.is_none());

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn direct_friendly_sql_native_aggregate_parity_functions_lower() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-aggregate-parity");
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
    assert!(source.reader_predicate.is_none());
    assert_eq!(
        source.read_columns,
        vec!["key".to_string(), "m00".to_string(), "m01".to_string(), "payload".to_string()]
    );

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn direct_friendly_sql_statistical_aggregate_aliases_lower() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-statistical-aggregates");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let plan = sql_plan(&path, direct_friendly_sql_session_config(), SQL_NATIVE_STATISTICAL_AGGREGATE_QUERY).await;

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.candidate_count(), 1);
    assert_eq!(result.rewritten_count(), 1);
    assert_eq!(result.rejected_count(), 0);
    assert_eq!(
        native_aggregate_functions(&result.plans[0]),
        vec![
            AggregateFunction::StddevPop,
            AggregateFunction::VarSamp,
            AggregateFunction::VarPop,
            AggregateFunction::VarSamp,
        ]
    );

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn direct_physical_count_distinct_lowers_to_exact_nunique() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-count-distinct-nunique");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let plan = sql_benchmark_plan(&path, direct_friendly_sql_session_config()).await;
    let sort = plan.downcast_ref::<SortExec>().expect("direct SQL root should be SortExec");
    let projection =
        sort.input().downcast_ref::<ProjectionExec>().expect("direct SQL sort input should be ProjectionExec");
    let aggregate = projection
        .input()
        .downcast_ref::<AggregateExec>()
        .expect("direct SQL projection input should be AggregateExec");
    let m01_index = aggregate.input().schema().index_of("m01").expect("aggregate input should include m01");
    let distinct_count = aggregate_expr_on_column_with_alias(
        count_udaf(),
        aggregate.input().schema(),
        m01_index,
        "distinct_payload",
        true,
        vec![],
    );
    let aggregate: Arc<dyn ExecutionPlan> = Arc::new(aggregate.with_new_aggr_exprs(vec![distinct_count]));

    let result = lower_physical_plan_to_native(aggregate);

    assert_eq!(result.candidate_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.rewritten_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.rejected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let aggregate_exprs = native_aggregate_exprs(&result.plans[0]);
    assert_eq!(aggregate_exprs.len(), 1);
    assert_eq!(aggregate_exprs[0].function, AggregateFunction::NUnique);
    assert!(!aggregate_exprs[0].distinct);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn direct_friendly_sql_count_star_lowers_with_synthetic_argument_column() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-count-star");
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
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-zero-column-count");
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
async fn direct_friendly_sql_typed_null_count_does_not_use_zero_column_count() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-count-typed-null");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let plan = typed_null_count_only_sort_limit_plan(parquet_scan(&path));

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.candidate_count(), 1);
    assert_eq!(result.rewritten_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.rejected_count(), 0, "outcomes={:?}", result.report.outcomes());
    assert!(
        !result
            .source_diagnostics
            .iter()
            .any(|diagnostics| { diagnostics.zero_column_count_disposition.as_str() == "represented" })
    );
    let source = native_local_parquet_source(&result.plans[0]);
    assert!(!source.is_zero_column_count_source());
    let aggregate_exprs = native_aggregate_exprs(&result.plans[0]);
    assert_eq!(aggregate_exprs.len(), 1);
    assert_eq!(aggregate_exprs[0].function, AggregateFunction::Count);
    assert!(matches!(
        &aggregate_exprs[0].expr,
        Expr::Column(column) if column.name().starts_with("__cudf_aggregate_arg_")
    ));

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn sql_count_only_lowers_to_native_zero_column_count_without_manual_plan_shape() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-sql-count-only");
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
            candidate: datafusion_nexus::GpuCandidate::with_operator_count(
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
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-sql-filter-rewrite");
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
    assert_eq!(result.source_diagnostics[0].predicate_disposition.as_str(), "native_post_scan_only");
    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(
        source.read_columns,
        vec!["key".to_string(), "m00".to_string(), "m01".to_string(), "payload".to_string()]
    );

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn direct_friendly_sql_not_filter_lowers_to_native_expr() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-sql-not-filter");
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
