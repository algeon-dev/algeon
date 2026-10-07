use super::*;
use crate::native::ParquetReadStrategyConfig;

#[tokio::test]
async fn default_sql_physical_plan_lowers_to_native_plan() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-sql-default");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let plan = sql_benchmark_plan(&path, SessionConfig::new()).await;

    let result = lower_physical_plan_to_native(plan, &ParquetReadStrategyConfig::default()).unwrap();

    assert_eq!(result.candidate_count(), 1);
    assert_eq!(result.selected_count(), 1);
    assert_eq!(result.not_selected_count(), 0);
    let CandidateOutcome::Selected { candidate } = &result.report.outcomes()[0] else {
        panic!("default SQL plan should lower to native");
    };
    assert_eq!(
        candidate.shape.to_string(),
        "SortPreservingMergeExec(fetch)(ProjectionExec(SortExec(FinalAggregateExec(RepartitionExec(PartialAggregateExec(FilterExec(LocalParquetScan)))))))"
    );
    assert_eq!(candidate.operator_count(), 8);
    assert!(result.plan.is_some());
    assert_eq!(result.source_diagnostics.len(), 1);
    assert_eq!(result.source_diagnostics[0].file_count, 1);
    assert!(result.source_diagnostics[0].direct_device_read);
    assert_eq!(result.source_diagnostics[0].predicate_disposition.as_str(), "pruning_only");
    assert!(result.source_diagnostics[0].rejected_features.is_empty());
    let source = native_local_parquet_source(result.selected_plan());
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
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-distributed-avg");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let plan = sql_plan(&path, SessionConfig::new(), SQL_AVG_AGGREGATE_QUERY).await;

    let result = lower_physical_plan_to_native(plan, &ParquetReadStrategyConfig::default()).unwrap();

    assert_eq!(result.candidate_count(), 1);
    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let aggregate_exprs = native_aggregate_exprs(result.selected_plan());
    assert_eq!(aggregate_exprs.len(), 1);
    assert_eq!(aggregate_exprs[0].function, AggregateFunction::Avg);
    assert_eq!(aggregate_exprs[0].alias, "avg(bench.m00)");
    assert!(
        matches!(&aggregate_exprs[0].expr, Expr::Column(column) if column.name() == "__algeon_native_aggregate_arg_0")
    );
    assert!(native_projection_exprs(result.selected_plan()).iter().any(|expr| matches!(
        expr,
        Expr::Alias { expr, alias }
            if alias == "__algeon_native_aggregate_arg_0"
                && matches!(
                    expr.as_ref(),
                    Expr::Cast { expr, target_type }
                        if target_type == &DataType::Float64
                            && matches!(expr.as_ref(), Expr::Column(column) if column.name() == "m00")
                )
    )));
    assert_eq!(
        result.selected_plan().output_schema().field_with_name("avg_m00").unwrap().data_type(),
        &DataType::Float64
    );
    let source = native_local_parquet_source(result.selected_plan());
    assert_eq!(source.read_columns, vec!["key".to_string(), "m00".to_string(), "payload".to_string()]);
    assert!(source.diagnostics.direct_device_read);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn default_sql_distributed_median_aggregate_remains_rejected() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-distributed-median");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let plan = sql_plan(&path, SessionConfig::new(), SQL_MEDIAN_AGGREGATE_QUERY).await;

    let result = lower_physical_plan_to_native(plan, &ParquetReadStrategyConfig::default()).unwrap();

    assert!(result.plan.is_none(), "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert!(
        matches!(
            result.report.outcomes()[0],
            CandidateOutcome::NotSupported {
                capability_reason: CapabilityReason::NativeDistributedAggregateUnsupported,
                ..
            }
        ),
        "outcomes={:?}",
        result.report.outcomes()
    );

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn default_sql_distributed_rollup_aggregate_lowers_to_native_grouping_sets() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-distributed-rollup");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let plan = sql_plan(&path, SessionConfig::new(), SQL_ROLLUP_AVG_QUERY).await;

    let result = lower_physical_plan_to_native(plan, &ParquetReadStrategyConfig::default()).unwrap();

    assert_eq!(result.candidate_count(), 1);
    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let aggregates = native_aggregate_nodes(result.selected_plan());
    assert_eq!(aggregates.len(), 1);
    let grouping_sets = aggregates[0].spec.grouping_sets.as_ref().expect("ROLLUP should lower as grouping sets");
    assert_eq!(grouping_sets.grouping_id_alias, "__grouping_id");
    assert_eq!(grouping_sets.groups, vec![vec![true, true], vec![false, true], vec![false, false]]);
    assert_eq!(aggregates[0].spec.group_exprs.len(), 2);
    assert_eq!(aggregates[0].spec.aggr_exprs.len(), 1);
    assert_eq!(aggregates[0].spec.aggr_exprs[0].function, AggregateFunction::Avg);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn grouping_sets_over_empty_local_parquet_lower_through_empty_literal_source() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-empty-grouping-sets");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);

    for (label, sql) in
        [("rollup", SQL_ROLLUP_AVG_QUERY), ("cube", SQL_CUBE_AVG_QUERY), ("grouping_sets", SQL_GROUPING_SETS_AVG_QUERY)]
    {
        let plan = sql_plan(&path, direct_friendly_sql_session_config(), sql).await;
        let plan = replace_parquet_scans_with_empty_file_groups(plan);

        let result = lower_physical_plan_to_native(plan, &ParquetReadStrategyConfig::default()).unwrap();

        assert_eq!(result.candidate_count(), 1, "{label}");
        assert_eq!(result.selected_count(), 1, "{label}: outcomes={:?}", result.report.outcomes());
        assert_eq!(result.not_selected_count(), 0, "{label}: outcomes={:?}", result.report.outcomes());
        assert!(native_local_parquet_sources(result.selected_plan()).is_empty(), "{label}");
        let literal_sources = native_literal_batch_sources(result.selected_plan());
        assert_eq!(literal_sources.len(), 1, "{label}");
        assert_eq!(literal_sources[0].row_count(), 0, "{label}");
        assert!(
            native_aggregate_nodes(result.selected_plan())
                .iter()
                .all(|aggregate| aggregate.spec.grouping_sets.is_some()),
            "{label} must retain native grouping-set semantics"
        );
    }

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn default_sql_distributed_aggregate_filter_lowers_to_native_filter() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-distributed-filtered-aggregate");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let plan = sql_plan(&path, SessionConfig::new(), SQL_FILTERED_AGGREGATE_QUERY).await;

    let result = lower_physical_plan_to_native(plan, &ParquetReadStrategyConfig::default()).unwrap();

    assert_eq!(result.candidate_count(), 1);
    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let aggregate_exprs = native_aggregate_exprs(result.selected_plan());
    assert_eq!(aggregate_exprs.len(), 1);
    assert_eq!(aggregate_exprs[0].function, AggregateFunction::Sum);
    assert!(
        matches!(
            aggregate_exprs[0].filter.as_ref(),
            Some(Expr::Column(column)) if column.name() == "__algeon_native_aggregate_filter_0"
        ),
        "distributed aggregate filter should be materialized once before the native aggregate"
    );
    assert!(native_projection_exprs(result.selected_plan()).iter().any(|expr| matches!(
        expr,
        Expr::Alias { alias, .. } if alias == "__algeon_native_aggregate_filter_0"
    )));

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn default_sql_distributed_distinct_aggregate_lowers_as_explicit_aggregate_stack() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-distributed-distinct-aggregate");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let plan = sql_plan(&path, SessionConfig::new(), SQL_DISTINCT_AGGREGATE_QUERY).await;

    let result = lower_physical_plan_to_native(plan, &ParquetReadStrategyConfig::default()).unwrap();

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    assert!(
        native_aggregate_nodes(result.selected_plan()).len() >= 2,
        "distinct COUNT should be represented by DataFusion's explicit aggregate stack"
    );

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn default_sql_mixed_distinct_and_count_over_union_lowers_to_one_native_aggregate() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-mixed-distinct-count-union");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let plan = sql_plan(&path, SessionConfig::new(), SQL_MIXED_DISTINCT_COUNT_UNION_QUERY).await;

    let result = lower_physical_plan_to_native(plan, &ParquetReadStrategyConfig::default()).unwrap();

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let aggregates = native_aggregate_nodes(result.selected_plan());
    assert_eq!(aggregates.len(), 1);
    assert_eq!(
        aggregates[0].spec.aggr_exprs.iter().map(|expr| (expr.function, expr.distinct)).collect::<Vec<_>>(),
        vec![(AggregateFunction::NUnique, false), (AggregateFunction::Count, false)]
    );

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn distributed_final_partial_aggregate_function_mismatch_is_rejected() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-distributed-function-mismatch");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let plan = sql_benchmark_plan(&path, SessionConfig::new()).await;
    let plan = with_rewritten_distributed_final_aggregate(plan, |final_aggregate| {
        let original = &final_aggregate.aggr_expr()[0];
        aggregate_expr_on_column_like(original, max_udaf(), final_aggregate.input().schema(), 1, false, vec![])
    });

    let result = lower_physical_plan_to_native(plan, &ParquetReadStrategyConfig::default()).unwrap();

    assert_distributed_aggregate_rejection(&result, CapabilityReason::NativeDistributedAggregateUnsupported);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn distributed_final_aggregate_state_schema_mismatch_is_rejected() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-distributed-state-schema-mismatch");
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

    let result = lower_physical_plan_to_native(plan, &ParquetReadStrategyConfig::default()).unwrap();

    assert_distributed_aggregate_rejection(&result, CapabilityReason::NativeDistributedAggregateUnsupported);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn distributed_partial_aggregate_filter_lowers_to_native_filter() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-distributed-filter-mutation");
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

    let result = lower_physical_plan_to_native(plan, &ParquetReadStrategyConfig::default()).unwrap();

    assert_eq!(result.candidate_count(), 1);
    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    assert_eq!(result.not_selected_count(), 0, "outcomes={:?}", result.report.outcomes());
    let aggregate_exprs = native_aggregate_exprs(result.selected_plan());
    assert_eq!(aggregate_exprs.len(), 1);
    assert!(
        matches!(
            aggregate_exprs[0].filter.as_ref(),
            Some(Expr::Column(column)) if column.name() == "__algeon_native_aggregate_filter_0"
        ),
        "partial-side filter should be retained on the native aggregate"
    );

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn distributed_final_aggregate_filter_expr_is_rejected() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-distributed-final-filter-mutation");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let plan = sql_benchmark_plan(&path, SessionConfig::new()).await;
    let plan = with_rewritten_distributed_final_aggregate_exec(plan, |final_aggregate| {
        let input_schema = final_aggregate.input().schema();
        let predicate = binary(
            col("key", &input_schema).unwrap(),
            Operator::GtEq,
            lit(ScalarValue::Int32(Some(10))),
            &input_schema,
        )
        .unwrap();
        Arc::new(
            AggregateExec::try_new(
                *final_aggregate.mode(),
                final_aggregate.group_expr().clone(),
                final_aggregate.aggr_expr().to_vec(),
                vec![Some(predicate)],
                Arc::clone(final_aggregate.input()),
                final_aggregate.input_schema(),
            )
            .unwrap(),
        )
    });

    let result = lower_physical_plan_to_native(plan, &ParquetReadStrategyConfig::default()).unwrap();

    assert_distributed_aggregate_rejection(&result, CapabilityReason::NativeDistributedAggregateUnsupported);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn distributed_aggregate_distinct_expr_is_rejected() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-distributed-distinct-mutation");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let plan = sql_benchmark_plan(&path, SessionConfig::new()).await;
    let plan = with_rewritten_distributed_partial_aggregate(plan, |partial_aggregate| {
        let original = &partial_aggregate.aggr_expr()[0];
        let distinct = aggregate_expr_like(original, sum_udaf(), partial_aggregate.input().schema(), true, vec![]);
        Arc::new(partial_aggregate.with_new_aggr_exprs(vec![distinct]))
    });

    let result = lower_physical_plan_to_native(plan, &ParquetReadStrategyConfig::default()).unwrap();

    assert_distributed_aggregate_rejection(&result, CapabilityReason::NativeDistributedAggregateUnsupported);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn distributed_aggregate_order_by_expr_is_rejected() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-distributed-order-by-mutation");
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

    let result = lower_physical_plan_to_native(plan, &ParquetReadStrategyConfig::default()).unwrap();

    assert_distributed_aggregate_rejection(&result, CapabilityReason::NativeDistributedAggregateUnsupported);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn distributed_aggregate_without_partial_source_shape_is_rejected() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-distributed-unexpected-source");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let plan = sql_benchmark_plan(&path, SessionConfig::new()).await;
    let plan = with_distributed_final_input_replaced_by_scan(plan);

    let result = lower_physical_plan_to_native(plan, &ParquetReadStrategyConfig::default()).unwrap();

    assert_distributed_aggregate_rejection(&result, CapabilityReason::NativeDistributedAggregateUnsupported);
    assert_eq!(
        rejection_blocking_text(&result),
        "distributed aggregate where the final stage reads DataSourceExec instead of a partial stage"
    );

    fs::remove_dir_all(dir).unwrap();
}

/// A source that declares its ordering makes DataFusion stream the final
/// aggregate over sorted groups, and the partial aggregate below a `UnionExec`
/// produces one sorted run per branch. DataFusion merges the runs with a
/// `SortPreservingMergeExec` between the two stages. The merge only
/// interleaves rows, so the pair lowers as it does behind a repartition,
/// whether the aggregate computes state or only deduplicates group keys.
#[tokio::test]
async fn distributed_aggregate_over_merge_of_sorted_partitions_lowers_to_native_aggregate() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-distributed-sorted-merge");
    write_sorted_numeric_parquet_file_with_sorting_columns(&dir.join("input.parquet"));

    for (sql, aggregate_function_count) in [
        ("SELECT key FROM bench UNION SELECT key FROM bench", 0),
        (
            "SELECT a.key AS v FROM bench a JOIN bench b ON a.key = b.key \
             UNION SELECT b.key AS v FROM bench a JOIN bench b ON a.key = b.key",
            0,
        ),
        (
            "SELECT key, SUM(m00) AS total FROM \
             (SELECT key, m00 FROM bench UNION ALL SELECT key, m00 FROM bench) GROUP BY key",
            1,
        ),
    ] {
        let plan = sql_plan(&dir, direct_friendly_sql_session_config(), sql).await;

        let result = lower_physical_plan_to_native(plan, &ParquetReadStrategyConfig::default()).unwrap();

        assert_eq!(result.selected_count(), 1, "{sql}\noutcomes={:?}", result.report.outcomes());
        let CandidateOutcome::Selected { candidate } = &result.report.outcomes()[0] else {
            panic!("{sql}\nthe sorted-merge aggregate should lower to native");
        };
        assert!(
            candidate.shape.to_string().contains("FinalAggregateExec(SortPreservingMergeExec(PartialAggregateExec("),
            "{sql}\nshape={}",
            candidate.shape
        );
        let aggregates = native_aggregate_nodes(result.selected_plan());
        assert_eq!(aggregates.len(), 1, "{sql}");
        assert_eq!(aggregates[0].spec.aggr_exprs.len(), aggregate_function_count, "{sql}");
    }

    fs::remove_dir_all(dir).unwrap();
}

/// A merge with a fetch drops rows, so the final aggregate would read fewer
/// rows than its partial stage produced. The pair stays rejected and the
/// blocker names the merge. The plan is already one partition, and the union
/// below the partial stage keeps its branches as separate partitions, so the
/// remedy is an engine gap and not planning for one partition.
#[tokio::test]
async fn distributed_aggregate_over_merge_with_fetch_is_rejected() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-distributed-sorted-merge-fetch");
    write_sorted_numeric_parquet_file_with_sorting_columns(&dir.join("input.parquet"));
    let plan =
        sql_plan(&dir, direct_friendly_sql_session_config(), "SELECT key FROM bench UNION SELECT key FROM bench").await;
    let merge = Arc::clone(plan.children()[0]);
    assert!(merge.is::<SortPreservingMergeExec>(), "the final aggregate should read a merge: {merge:?}");
    let limited_merge = merge.with_fetch(Some(1)).unwrap();
    let plan = plan
        .replace_children(vec![limited_merge], ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute))
        .unwrap();

    let result = lower_physical_plan_to_native(plan, &ParquetReadStrategyConfig::default()).unwrap();

    assert_distributed_aggregate_rejection(&result, CapabilityReason::NativeDistributedAggregateUnsupported);
    assert_eq!(
        rejection_blocking_text(&result),
        "distributed aggregate where the final stage reads SortPreservingMergeExec instead of a partial stage"
    );
    assert_eq!(rejected_candidate_remedy(&result).code, "unsupported_operator_gap");

    fs::remove_dir_all(dir).unwrap();
}

fn rejection_blocking_text(result: &NativeLoweringResult) -> String {
    let CandidateOutcome::NotSupported { detail, .. } = &result.report.outcomes()[0] else {
        panic!("expected rejected native candidate");
    };
    let blocking = detail.blocking().expect("the rejection names its blocker");
    assert_eq!(blocking.kind(), "distributed_aggregate");
    blocking.text().to_string()
}
