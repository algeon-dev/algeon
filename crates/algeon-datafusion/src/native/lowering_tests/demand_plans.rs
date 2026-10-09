use datafusion::physical_plan::joins::CrossJoinExec;

use super::*;
use crate::native::ParquetReadStrategyConfig;

fn scalar_aggregate(
    scan: Arc<dyn ExecutionPlan>,
    function: Arc<AggregateUDF>,
    argument: Arc<dyn PhysicalExpr>,
    alias: &str,
) -> Arc<dyn ExecutionPlan> {
    let scan_schema = scan.schema();
    let aggregate_expr = AggregateExprBuilder::new(function, vec![argument])
        .schema(Arc::clone(&scan_schema))
        .alias(alias)
        .build()
        .unwrap();
    Arc::new(
        AggregateExec::try_new(
            AggregateMode::Single,
            PhysicalGroupBy::new_single(Vec::new()),
            vec![Arc::new(aggregate_expr)],
            vec![None],
            scan,
            scan_schema,
        )
        .unwrap(),
    )
}

#[test]
fn one_scan_under_two_demands_lowers_once_per_demand() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-shared-scan-demands");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let scan = parquet_scan(&path);
    let scan_schema = scan.schema();
    // The same scan node reads only its row count for COUNT(*) and the m00
    // column for SUM(m00).
    let count = scalar_aggregate(Arc::clone(&scan), count_udaf(), lit(ScalarValue::Int64(Some(1))), "count_rows");
    let sum = scalar_aggregate(scan, sum_udaf(), col("m00", &scan_schema).unwrap(), "sum_m00");
    let plan: Arc<dyn ExecutionPlan> = Arc::new(CrossJoinExec::new(count, sum));

    let result = lower_physical_plan_to_native(plan, &ParquetReadStrategyConfig::default()).unwrap();

    assert_eq!(result.selected_count(), 1, "outcomes={:?}", result.report.outcomes());
    let sources = result
        .selected_plan()
        .nodes()
        .iter()
        .filter_map(|node| match &node.kind {
            PlanNode::Source(source) => match source.as_ref() {
                SourceNode::LocalParquet(source) => Some(source),
                _ => None,
            },
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(sources.len(), 2, "{}", result.selected_plan());
    assert!(sources[0].is_zero_column_count_source(), "COUNT(*) reads no column: {}", result.selected_plan());
    assert_eq!(sources[1].read_columns, vec!["m00".to_string()], "SUM(m00) reads m00: {}", result.selected_plan());

    fs::remove_dir_all(dir).unwrap();
}

/// Native plans for a `COUNT(*)`, a filtered aggregate and a top-k query, as
/// `QueryPlan` displays them.
#[tokio::test]
async fn representative_query_plans_match_golden() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-representative-plans");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    // Without file statistics DataFusion keeps COUNT(*) and does not fold it
    // to a literal.
    let config = || SessionConfig::new().with_target_partitions(1).with_collect_statistics(false);
    let mut displays = Vec::new();
    for sql in [
        "SELECT COUNT(*) FROM bench",
        "SELECT key, SUM(m00 + m01) AS total FROM bench WHERE payload > 5 GROUP BY key",
        "SELECT key, m00 FROM bench ORDER BY m00 DESC LIMIT 2",
    ] {
        let plan = sql_plan(&path, config(), sql).await;
        let result = lower_physical_plan_to_native(plan, &ParquetReadStrategyConfig::default()).unwrap();
        assert_eq!(result.selected_count(), 1, "{sql}: outcomes={:?}", result.report.outcomes());
        let display = result.selected_plan().to_string().replace(dir.to_str().unwrap(), "<dir>");
        displays.push(format!("-- {sql}\n{display}"));
    }
    let actual = displays.join("\n") + "\n";
    assert_eq!(actual, include_str!("goldens/representative_query_plans.txt"));

    fs::remove_dir_all(dir).unwrap();
}
