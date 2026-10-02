use query_engine::{
    policy::PlanningPolicy, rapids_backend::rapids_analysis_context, source::validate_local_parquet_source,
};

use super::*;
use crate::native::ParquetReadStrategyConfig;

#[tokio::test]
async fn direct_friendly_sql_physical_plan_has_stable_native_lowering_outcome() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-sql-friendly");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let plan = sql_benchmark_plan(&path, direct_friendly_sql_session_config()).await;

    let result = lower_physical_plan_to_native(plan, &ParquetReadStrategyConfig::default()).unwrap();

    assert_eq!(result.candidate_count(), 1);
    assert_eq!(result.selected_count(), 1);
    assert_eq!(result.not_selected_count(), 0);
    let CandidateOutcome::Selected { candidate } = &result.report.outcomes()[0] else {
        panic!("direct-friendly SQL plan should lower to native");
    };
    assert_eq!(
        candidate.shape.to_string(),
        "ProjectionExec(SortExec(fetch)(AggregateExec(FilterExec(LocalParquetScan))))"
    );
    assert_eq!(candidate.operator_count(), 5);
    assert!(result.plan.is_some());
    assert_eq!(result.source_diagnostics.len(), 1);
    assert_eq!(result.source_diagnostics[0].file_count, 1);
    assert!(result.source_diagnostics[0].direct_device_read);
    assert_eq!(result.source_diagnostics[0].predicate_disposition.as_str(), "pruning_only");
    assert!(result.source_diagnostics[0].rejected_features.is_empty());
    let source = native_local_parquet_source(result.selected_plan());
    assert!(source.reader_predicate.is_none());

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn direct_friendly_sql_native_aggregate_parity_functions_lower() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-aggregate-parity");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let plan = sql_plan(&path, direct_friendly_sql_session_config(), SQL_NATIVE_AGGREGATE_PARITY_QUERY).await;

    let result = lower_physical_plan_to_native(plan, &ParquetReadStrategyConfig::default()).unwrap();

    assert_eq!(result.candidate_count(), 1);
    assert_eq!(result.selected_count(), 1);
    assert_eq!(result.not_selected_count(), 0);
    let functions = native_aggregate_functions(result.selected_plan());
    assert_eq!(functions, vec![AggregateFunction::Min, AggregateFunction::Max, AggregateFunction::Count]);
    assert_eq!(result.source_diagnostics.len(), 1);
    assert!(result.source_diagnostics[0].direct_device_read);
    assert_eq!(result.source_diagnostics[0].predicate_disposition.as_str(), "pruning_only");
    assert!(result.source_diagnostics[0].rejected_features.is_empty());
    let source = native_local_parquet_source(result.selected_plan());
    assert!(source.reader_predicate.is_none());
    assert_eq!(
        source.read_columns,
        vec!["key".to_string(), "m00".to_string(), "m01".to_string(), "payload".to_string()]
    );

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn direct_friendly_sql_median_aggregate_lowers_to_native_median() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-median");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let plan = sql_plan(&path, direct_friendly_sql_session_config(), SQL_MEDIAN_AGGREGATE_QUERY).await;

    let result = lower_physical_plan_to_native(plan, &ParquetReadStrategyConfig::default()).unwrap();

    assert_eq!(result.candidate_count(), 1);
    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let aggregate_exprs = native_aggregate_exprs(result.selected_plan());
    assert_eq!(aggregate_exprs.len(), 1);
    assert_eq!(aggregate_exprs[0].function, AggregateFunction::Median);
    assert_eq!(aggregate_exprs[0].alias, "median(bench.m00)");

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn direct_friendly_sql_mixed_median_and_statistical_aggregate_is_rejected() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-mixed-median-statistical");
    let path = dir.join("input.parquet");
    write_float_measure_parquet_file(&path);
    let plan = sql_plan(
        &path,
        direct_friendly_sql_session_config(),
        "SELECT key, STDDEV_SAMP(m00) AS stdev_m00, MEDIAN(m00) AS median_m00 FROM bench GROUP BY key",
    )
    .await;

    let result = lower_physical_plan_to_native(plan, &ParquetReadStrategyConfig::default()).unwrap();

    assert_native_rejection(&result, CapabilityReason::UnsupportedOperator);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn direct_friendly_sql_statistical_aggregate_aliases_lower() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-statistical-aggregates");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let plan = sql_plan(&path, direct_friendly_sql_session_config(), SQL_NATIVE_STATISTICAL_AGGREGATE_QUERY).await;

    let result = lower_physical_plan_to_native(plan, &ParquetReadStrategyConfig::default()).unwrap();

    assert_eq!(result.candidate_count(), 1);
    assert_eq!(result.selected_count(), 1);
    assert_eq!(result.not_selected_count(), 0);
    assert_eq!(
        native_aggregate_functions(result.selected_plan()),
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
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-count-distinct-nunique");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let plan = sql_benchmark_plan(&path, direct_friendly_sql_session_config()).await;
    let aggregate = if let Some(proj) = plan.downcast_ref::<ProjectionExec>() {
        let sort = proj.input().downcast_ref::<SortExec>().expect("direct SQL projection input should be SortExec");
        sort.input().downcast_ref::<AggregateExec>().expect("direct SQL sort input should be AggregateExec")
    } else {
        let sort = plan.downcast_ref::<SortExec>().expect("direct SQL root should be SortExec");
        let proj =
            sort.input().downcast_ref::<ProjectionExec>().expect("direct SQL sort input should be ProjectionExec");
        proj.input().downcast_ref::<AggregateExec>().expect("direct SQL projection input should be AggregateExec")
    };
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

    let result = lower_physical_plan_to_native(aggregate, &ParquetReadStrategyConfig::default()).unwrap();

    assert_eq!(result.candidate_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let aggregate_exprs = native_aggregate_exprs(result.selected_plan());
    assert_eq!(aggregate_exprs.len(), 1);
    assert_eq!(aggregate_exprs[0].function, AggregateFunction::NUnique);
    assert!(!aggregate_exprs[0].distinct);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn direct_friendly_sql_count_star_lowers_with_synthetic_argument_column() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-count-star");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let plan = sql_plan(&path, direct_friendly_sql_session_config(), SQL_COUNT_STAR_QUERY).await;

    let result = lower_physical_plan_to_native(plan, &ParquetReadStrategyConfig::default()).unwrap();

    assert_eq!(result.selected_count(), 1);
    assert_eq!(result.not_selected_count(), 0);
    let aggregate_exprs = native_aggregate_exprs(result.selected_plan());
    assert_eq!(aggregate_exprs.len(), 2);
    assert_eq!(
        native_aggregate_functions(result.selected_plan()),
        vec![AggregateFunction::Sum, AggregateFunction::Count]
    );
    let count_expr = aggregate_exprs
        .iter()
        .find(|expr| expr.function == AggregateFunction::Count)
        .expect("native aggregate should include COUNT(*)");
    assert!(matches!(
        &count_expr.expr,
        Expr::Column(column) if column.name().starts_with("__algeon_native_aggregate_arg_")
    ));
    let source = native_local_parquet_source(result.selected_plan());
    assert_eq!(
        source.read_columns,
        vec!["key".to_string(), "m00".to_string(), "m01".to_string(), "payload".to_string()]
    );
    assert!(source.diagnostics.direct_device_read);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn direct_friendly_sql_zero_column_count_lowers_to_represented_source() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-zero-column-count");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let plan = count_only_sort_limit_plan(parquet_scan(&path));

    let result = lower_physical_plan_to_native(plan, &ParquetReadStrategyConfig::default()).unwrap();

    assert_eq!(result.candidate_count(), 1);
    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.source_diagnostics.len(), 1);
    assert!(result.source_diagnostics[0].direct_device_read);
    assert_eq!(result.source_diagnostics[0].read_column_count, 0);
    assert_eq!(result.source_diagnostics[0].projected_output_column_count, 0);
    assert_eq!(result.source_diagnostics[0].pure_filter_column_count, 0);
    assert_eq!(result.source_diagnostics[0].zero_column_count_disposition.as_str(), "represented");
    assert!(result.source_diagnostics[0].rejected_features.is_empty());

    let source = native_local_parquet_source(result.selected_plan());
    assert!(source.is_zero_column_count_source());
    assert!(source.read_columns.is_empty());
    assert!(source.read_column_file_indices.is_empty());
    assert_eq!(source.diagnostics.zero_column_count_disposition.as_str(), "represented");
    let aggregate_exprs = native_aggregate_exprs(result.selected_plan());
    assert_eq!(aggregate_exprs.len(), 1);
    assert_eq!(aggregate_exprs[0].function, AggregateFunction::Sum);
    assert!(matches!(&aggregate_exprs[0].expr, Expr::Column(column) if column.name() == ZERO_COLUMN_COUNT_COLUMN));

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn direct_friendly_sql_typed_null_count_does_not_use_zero_column_count() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-count-typed-null");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let plan = typed_null_count_only_sort_limit_plan(parquet_scan(&path));

    let result = lower_physical_plan_to_native(plan, &ParquetReadStrategyConfig::default()).unwrap();

    assert_eq!(result.candidate_count(), 1);
    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    assert!(
        !result
            .source_diagnostics
            .iter()
            .any(|diagnostics| { diagnostics.zero_column_count_disposition.as_str() == "represented" })
    );
    let source = native_local_parquet_source(result.selected_plan());
    assert!(!source.is_zero_column_count_source());
    assert_eq!(source.read_columns, vec!["key".to_string()]);
    assert_eq!(source.projected_output_columns, Vec::<String>::new());
    assert_eq!(source.pure_filter_columns, vec!["key".to_string()]);
    assert_eq!(source.diagnostics.read_column_count, 1);
    assert_eq!(source.diagnostics.projected_output_column_count, 0);
    assert_eq!(source.diagnostics.pure_filter_column_count, 1);
    assert_eq!(source.diagnostics.zero_column_count_disposition.as_str(), "not_required");
    validate_local_parquet_source(source, rapids_analysis_context(PlanningPolicy::default())).unwrap();
    let aggregate_exprs = native_aggregate_exprs(result.selected_plan());
    assert_eq!(aggregate_exprs.len(), 1);
    assert_eq!(aggregate_exprs[0].function, AggregateFunction::Count);
    assert!(matches!(
        &aggregate_exprs[0].expr,
        Expr::Column(column) if column.name().starts_with("__algeon_native_aggregate_arg_")
    ));

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn sql_count_only_lowers_to_native_zero_column_count_without_manual_plan_shape() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-sql-count-only");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let plan = sql_plan(&path, count_only_sql_session_config(), SQL_COUNT_ONLY_QUERY).await;

    let result = lower_physical_plan_to_native(plan, &ParquetReadStrategyConfig::default()).unwrap();

    assert_eq!(result.candidate_count(), 1);
    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let CandidateOutcome::Selected { candidate } = &result.report.outcomes()[0] else {
        panic!("outcomes={:?}", result.report.outcomes());
    };
    assert_eq!(candidate.shape.to_string(), "ProjectionExec(AggregateExec(LocalParquetScan))");
    assert_eq!(candidate.output_schema_fields, 1);
    assert_eq!(candidate.operator_count(), 3);
    assert_eq!(result.source_diagnostics.len(), 1);
    assert!(result.source_diagnostics[0].direct_device_read);
    assert_eq!(result.source_diagnostics[0].read_column_count, 0);
    assert_eq!(result.source_diagnostics[0].zero_column_count_disposition.as_str(), "represented");

    let source = native_local_parquet_source(result.selected_plan());
    assert!(source.is_zero_column_count_source());
    assert!(source.read_columns.is_empty());
    let aggregate_exprs = native_aggregate_exprs(result.selected_plan());
    assert_eq!(aggregate_exprs.len(), 1);
    assert_eq!(aggregate_exprs[0].function, AggregateFunction::Sum);
    assert!(matches!(&aggregate_exprs[0].expr, Expr::Column(column) if column.name() == ZERO_COLUMN_COUNT_COLUMN));
    assert!(matches!(result.selected_plan().root_node().kind, PlanNode::Projection(_)));

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn direct_friendly_sql_filter_plan_lowers_when_scan_predicate_is_absent() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-sql-filter-rewrite");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let plan = sql_benchmark_plan(&path, direct_friendly_sql_session_config()).await;
    let plan = sql_plan_without_scan_predicate(plan);

    let result = lower_physical_plan_to_native(plan, &ParquetReadStrategyConfig::default()).unwrap();

    assert_eq!(result.candidate_count(), 1);
    assert_eq!(result.selected_count(), 1);
    assert_eq!(result.not_selected_count(), 0);
    let CandidateOutcome::Selected { candidate } = &result.report.outcomes()[0] else {
        panic!("direct-friendly SQL filter plan should lower to native");
    };
    assert_eq!(
        candidate.shape.to_string(),
        "ProjectionExec(SortExec(fetch)(AggregateExec(FilterExec(LocalParquetScan))))"
    );
    assert_eq!(candidate.operator_count(), 5);
    assert!(result.plan.is_some());
    assert_eq!(result.selected_plan().node_count(), 7);
    assert_eq!(result.source_diagnostics.len(), 1);
    assert!(result.source_diagnostics[0].direct_device_read);
    assert_eq!(result.source_diagnostics[0].predicate_disposition.as_str(), "native_post_scan_only");
    let source = native_local_parquet_source(result.selected_plan());
    assert_eq!(
        source.read_columns,
        vec!["key".to_string(), "m00".to_string(), "m01".to_string(), "payload".to_string()]
    );

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn direct_friendly_sql_not_filter_lowers_to_native_expr() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-sql-not-filter");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let scan = parquet_scan(&path);
    let scan_schema = scan.schema();
    let payload_lt_10 =
        binary(col("payload", &scan_schema).unwrap(), Operator::Lt, lit(ScalarValue::Int32(Some(10))), &scan_schema)
            .unwrap();
    let predicate = Arc::new(NotExpr::new(payload_lt_10));
    let plan = aggregate_sort_limit_plan_with_filter(scan, predicate, Operator::Plus);

    let result = lower_physical_plan_to_native(plan, &ParquetReadStrategyConfig::default()).unwrap();

    assert_eq!(result.candidate_count(), 1);
    assert_eq!(result.selected_count(), 1);
    assert_eq!(result.not_selected_count(), 0);
    let filter = native_filter_expr(result.selected_plan());
    assert_eq!(filter.to_string(), "NOT (payload < 10)");
    let source = native_local_parquet_source(result.selected_plan());
    assert_eq!(
        source.read_columns,
        vec!["key".to_string(), "m00".to_string(), "m01".to_string(), "payload".to_string()]
    );

    fs::remove_dir_all(dir).unwrap();
}
