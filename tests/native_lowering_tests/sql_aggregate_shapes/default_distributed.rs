use super::*;

#[tokio::test]
async fn default_sql_physical_plan_lowers_to_native_plan() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-sql-default");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let plan = sql_benchmark_plan(&path, SessionConfig::new()).await;

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.candidate_count(), 1);
    assert_eq!(result.rewritten_count(), 1);
    assert_eq!(result.rejected_count(), 0);
    let CandidateOutcome::Rewritten { candidate } = &result.report.outcomes()[0] else {
        panic!("default SQL plan should lower to native");
    };
    assert_eq!(
        candidate.shape,
        "SortPreservingMergeExec(fetch)(SortExec(ProjectionExec(FinalAggregateExec(RepartitionExec(PartialAggregateExec(FilterExec(RepartitionExec(LocalParquetScan))))))))"
    );
    assert_eq!(candidate.operator_count, 9);
    assert_eq!(result.plans.len(), 1);
    assert_eq!(result.source_diagnostics.len(), 1);
    assert_eq!(result.source_diagnostics[0].file_count, 1);
    assert!(result.source_diagnostics[0].direct_device_read);
    assert_eq!(result.source_diagnostics[0].predicate_disposition.as_str(), "pruning_only");
    assert!(result.source_diagnostics[0].rejected_features.is_empty());
    let source = native_local_parquet_source(&result.plans[0]);
    assert!(source.reader_predicate.is_none());
    assert_eq!(
        source.read_columns,
        vec!["key".to_string(), "m00".to_string(), "m01".to_string(), "payload".to_string()]
    );
    assert!(source.diagnostics.direct_device_read);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn default_sql_distributed_avg_aggregate_lowers_to_native_avg() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-distributed-avg");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let plan = sql_plan(&path, SessionConfig::new(), SQL_AVG_AGGREGATE_QUERY).await;

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.candidate_count(), 1);
    assert_eq!(result.rewritten_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.rejected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let aggregate_exprs = native_aggregate_exprs(&result.plans[0]);
    assert_eq!(aggregate_exprs.len(), 1);
    assert_eq!(aggregate_exprs[0].function, AggregateFunction::Avg);
    assert_eq!(aggregate_exprs[0].alias, "avg(bench.m00)");
    assert!(matches!(&aggregate_exprs[0].expr, Expr::Column(column) if column.name() == "__cudf_aggregate_arg_0"));
    assert!(native_projection_exprs(&result.plans[0]).iter().any(|expr| matches!(
        expr,
        Expr::Alias { expr, alias }
            if alias == "__cudf_aggregate_arg_0"
                && matches!(
                    expr.as_ref(),
                    Expr::Cast { expr, target_type }
                        if target_type == &DataType::Float64
                            && matches!(expr.as_ref(), Expr::Column(column) if column.name() == "m00")
                )
    )));
    assert_eq!(result.plans[0].output_schema().field_with_name("avg_m00").unwrap().data_type(), &DataType::Float64);
    let source = native_local_parquet_source(&result.plans[0]);
    assert_eq!(source.read_columns, vec!["key".to_string(), "m00".to_string(), "payload".to_string()]);
    assert!(source.diagnostics.direct_device_read);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn default_sql_distributed_aggregate_filter_is_rejected() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-distributed-filtered-aggregate");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let plan = sql_plan(&path, SessionConfig::new(), SQL_FILTERED_AGGREGATE_QUERY).await;

    let result = lower_physical_plan_to_native(plan);

    assert_distributed_aggregate_rejection(&result, FallbackReason::NativeDistributedAggregateUnsupported);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn default_sql_distributed_distinct_aggregate_lowers_as_explicit_aggregate_stack() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-distributed-distinct-aggregate");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let plan = sql_plan(&path, SessionConfig::new(), SQL_DISTINCT_AGGREGATE_QUERY).await;

    let result = lower_physical_plan_to_native(plan);

    assert_eq!(result.rewritten_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.rejected_count(), 0, "outcomes={:?}", result.report.outcomes());
    assert!(
        native_aggregate_nodes(&result.plans[0]).len() >= 2,
        "distinct COUNT should be represented by DataFusion's explicit aggregate stack"
    );

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn distributed_final_partial_aggregate_function_mismatch_is_rejected() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-distributed-function-mismatch");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let plan = sql_benchmark_plan(&path, SessionConfig::new()).await;
    let plan = with_rewritten_distributed_final_aggregate(plan, |final_aggregate| {
        let original = &final_aggregate.aggr_expr()[0];
        aggregate_expr_on_column_like(original, max_udaf(), final_aggregate.input().schema(), 1, false, vec![])
    });

    let result = lower_physical_plan_to_native(plan);

    assert_distributed_aggregate_rejection(&result, FallbackReason::NativeDistributedAggregateUnsupported);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn distributed_final_aggregate_state_schema_mismatch_is_rejected() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-distributed-state-schema-mismatch");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let plan = sql_plan(&path, SessionConfig::new(), SQL_COUNT_STAR_QUERY).await;
    let plan = with_rewritten_distributed_final_aggregate(plan, |final_aggregate| {
        aggregate_expr_on_column_with_alias(
            sum_udaf(),
            final_aggregate.input().schema(),
            1,
            "count_rows",
            false,
            vec![],
        )
    });

    let result = lower_physical_plan_to_native(plan);

    assert_distributed_aggregate_rejection(&result, FallbackReason::NativeDistributedAggregateUnsupported);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn distributed_aggregate_filter_expr_is_rejected() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-distributed-filter-mutation");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let plan = sql_benchmark_plan(&path, SessionConfig::new()).await;
    let plan = with_rewritten_distributed_partial_aggregate(plan, |partial_aggregate| {
        let input_schema = partial_aggregate.input().schema();
        let predicate = binary(
            col("key", &input_schema).unwrap(),
            Operator::GtEq,
            lit(ScalarValue::Int32(Some(10))),
            &input_schema,
        )
        .unwrap();
        Arc::new(
            AggregateExec::try_new(
                *partial_aggregate.mode(),
                partial_aggregate.group_expr().clone(),
                partial_aggregate.aggr_expr().to_vec(),
                vec![Some(predicate)],
                Arc::clone(partial_aggregate.input()),
                partial_aggregate.input_schema(),
            )
            .unwrap(),
        )
    });

    let result = lower_physical_plan_to_native(plan);

    assert_distributed_aggregate_rejection(&result, FallbackReason::NativeDistributedAggregateUnsupported);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn distributed_aggregate_distinct_expr_is_rejected() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-distributed-distinct-mutation");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let plan = sql_benchmark_plan(&path, SessionConfig::new()).await;
    let plan = with_rewritten_distributed_partial_aggregate(plan, |partial_aggregate| {
        let original = &partial_aggregate.aggr_expr()[0];
        let distinct = aggregate_expr_like(original, sum_udaf(), partial_aggregate.input().schema(), true, vec![]);
        Arc::new(partial_aggregate.with_new_aggr_exprs(vec![distinct]))
    });

    let result = lower_physical_plan_to_native(plan);

    assert_distributed_aggregate_rejection(&result, FallbackReason::NativeDistributedAggregateUnsupported);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn distributed_aggregate_order_by_expr_is_rejected() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-distributed-order-by-mutation");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let plan = sql_benchmark_plan(&path, SessionConfig::new()).await;
    let plan = with_rewritten_distributed_partial_and_final_aggregate(
        plan,
        |partial_aggregate| {
            let input_schema = partial_aggregate.input().schema();
            let order_by =
                vec![PhysicalSortExpr::new(col("key", &input_schema).unwrap(), ArrowSortOptions::new(false, false))];
            let original = &partial_aggregate.aggr_expr()[0];
            let ordered = aggregate_expr_like(original, first_value_udaf(), input_schema, false, order_by);
            Arc::new(partial_aggregate.with_new_aggr_exprs(vec![ordered]))
        },
        |final_aggregate| {
            let input_schema = final_aggregate.input().schema();
            let order_by =
                vec![PhysicalSortExpr::new(col("key", &input_schema).unwrap(), ArrowSortOptions::new(false, false))];
            let original = &final_aggregate.aggr_expr()[0];
            aggregate_expr_on_column_like(original, first_value_udaf(), input_schema, 1, false, order_by)
        },
    );

    let result = lower_physical_plan_to_native(plan);

    assert_distributed_aggregate_rejection(&result, FallbackReason::NativeDistributedAggregateUnsupported);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn distributed_aggregate_without_partial_source_shape_is_rejected() {
    let dir = unique_temp_dir("datafusion-cudf-native-lowering-distributed-unexpected-source");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let plan = sql_benchmark_plan(&path, SessionConfig::new()).await;
    let plan = with_distributed_final_input_replaced_by_scan(plan);

    let result = lower_physical_plan_to_native(plan);

    assert_distributed_aggregate_rejection(&result, FallbackReason::NativeDistributedAggregateUnsupported);

    fs::remove_dir_all(dir).unwrap();
}
