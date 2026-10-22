use arrow::{array::Float64Array, record_batch::RecordBatch};
use datafusion::{datasource::memory::MemorySourceConfig, functions_aggregate::stddev::stddev_udaf};

use super::*;
use crate::{
    gpu_coverage::QueryGpuRemedyKind,
    native::{ParquetReadStrategyConfig, floating_statistical_aggregate_input_is_partitioned},
    report::{AggregateDefect, PlanBlocker},
};

/// Three files give a scan three file groups, so DataFusion plans partitioned
/// shapes when its target partition count allows them.
fn three_file_table(prefix: &str) -> std::path::PathBuf {
    let dir = unique_temp_dir(prefix);
    for index in 0..3 {
        write_numeric_parquet_file(&dir.join(format!("part{index}.parquet")));
    }
    dir
}

fn rejection_reason(result: &NativeLoweringResult) -> Option<CapabilityReason> {
    match result.report.outcomes() {
        [CandidateOutcome::NotSupported { capability_reason, .. }] => Some(*capability_reason),
        _ => None,
    }
}

/// Each shape below follows from DataFusion planning for several partitions,
/// and the config remedy of its candidate names the session setting that
/// avoids it. The last aggregate unions on the collected side of a join, which
/// DataFusion merges into one partition, so one partition removes its stages.
#[tokio::test]
async fn target_partitions_one_avoids_the_plan_shapes_whose_remedy_names_it() {
    let dir = three_file_table("algeon-datafusion-plan-shape-remedies");
    let cases = [
        (
            "SELECT key, MEDIAN(m00) AS median_m00 FROM bench GROUP BY key",
            CapabilityReason::NativeDistributedAggregateUnsupported,
        ),
        (
            "SELECT a.key, MEDIAN(a.m00) AS median_m00 FROM bench a \
             JOIN (SELECT key FROM bench UNION ALL SELECT key FROM bench) b ON a.key = b.key GROUP BY a.key",
            CapabilityReason::NativeDistributedAggregateUnsupported,
        ),
        (
            "SELECT key, COUNT(*) FROM bench GROUP BY key LIMIT 2",
            CapabilityReason::NativeLocalLimitPartitionUnsupported,
        ),
    ];
    for (sql, reason) in cases {
        let partitioned = sql_plan(&dir, SessionConfig::new().with_target_partitions(4), sql).await;
        let result = lower_physical_plan_to_native(partitioned, &ParquetReadStrategyConfig::default()).unwrap();
        assert_eq!(rejection_reason(&result), Some(reason), "{sql}");
        let remedy = rejected_candidate_remedy(&result);
        assert_eq!(remedy.code, "set_target_partitions_one", "{sql}");
        assert_eq!(remedy.kind, QueryGpuRemedyKind::ConfigAction, "{sql}");
        assert!(remedy.guidance.contains("SET datafusion.execution.target_partitions = 1"), "{sql}");

        let single = sql_plan(&dir, SessionConfig::new().with_target_partitions(1), sql).await;
        let result = lower_physical_plan_to_native(single, &ParquetReadStrategyConfig::default()).unwrap();
        assert_eq!(result.selected_count(), 1, "{sql}: {:?}", result.report.outcomes());
    }
    fs::remove_dir_all(dir).unwrap();
}

/// DataFusion keeps the partial and final stages of an aggregate at one
/// partition when the aggregate reads a `UNION ALL`, whose branches stay
/// separate partitions, on its probe side or directly, and when it groups by
/// grouping sets. A single-stage aggregate also rejects a function without a
/// native kernel and a median beside a standard deviation. In every case one
/// partition cannot lower the shape, so the candidate carries an engine gap
/// instead of the config remedy, at either partition count.
#[tokio::test]
async fn distributed_aggregate_remedy_is_an_engine_gap_when_one_partition_cannot_lower_it() {
    let dir = three_file_table("algeon-datafusion-distributed-aggregate-gap-remedy");
    let union_branches = "SELECT key, m00 FROM bench UNION ALL SELECT key, m00 FROM bench";
    let cases = [
        (
            format!("SELECT key, MEDIAN(m00) AS v FROM ({union_branches}) GROUP BY key"),
            CapabilityReason::NativeDistributedAggregateUnsupported,
        ),
        (
            format!(
                "SELECT a.key, MEDIAN(b.m00) AS v FROM bench a JOIN ({union_branches}) b ON a.key = b.key \
                 GROUP BY a.key"
            ),
            CapabilityReason::NativeDistributedAggregateUnsupported,
        ),
        (
            "SELECT key, STDDEV(m00) AS v FROM bench GROUP BY ROLLUP(key)".to_string(),
            CapabilityReason::NativeDistributedAggregateUnsupported,
        ),
        (
            "SELECT key, MEDIAN(m00) AS v, STDDEV(m01) AS s FROM bench GROUP BY key".to_string(),
            CapabilityReason::UnsupportedOperator,
        ),
        (
            "SELECT key, APPROX_DISTINCT(m00) AS v FROM bench GROUP BY key".to_string(),
            CapabilityReason::UnsupportedOperator,
        ),
    ];
    for (sql, reason_at_one_partition) in cases {
        for (target_partitions, expected_reason) in
            [(4, CapabilityReason::NativeDistributedAggregateUnsupported), (1, reason_at_one_partition)]
        {
            let config = SessionConfig::new().with_target_partitions(target_partitions);
            let plan = sql_plan(&dir, config, &sql).await;
            let result = lower_physical_plan_to_native(plan, &ParquetReadStrategyConfig::default()).unwrap();

            let context = format!("target_partitions = {target_partitions}: {sql}");
            assert_eq!(rejection_reason(&result), Some(expected_reason), "{context}");
            let remedy = rejected_candidate_remedy(&result);
            assert_eq!(remedy.code, "unsupported_operator_gap", "{context}");
            assert_eq!(remedy.kind, QueryGpuRemedyKind::EngineGap, "{context}");
        }
    }
    fs::remove_dir_all(dir).unwrap();
}

fn plan_has_expression_keyed_merge(plan: &dyn ExecutionPlan) -> bool {
    plan.downcast_ref::<SortPreservingMergeExec>()
        .is_some_and(|merge| merge.expr().iter().any(|sort_expr| sort_expr.expr.downcast_ref::<Column>().is_none()))
        || plan.children().into_iter().any(|child| plan_has_expression_keyed_merge(child.as_ref()))
}

/// One native sort by the merge's keys reproduces a merge of sorted
/// partitions, so a merge whose key is an expression the native sort accepts
/// lowers at any partition count. That holds when the merge reads `UNION ALL`
/// branches directly or through either side of a join. A key without a native
/// form is rejected as the expression it is at either partition count.
#[tokio::test]
async fn expression_keyed_merges_lower_at_any_partition_count() {
    let dir = three_file_table("algeon-datafusion-expression-keyed-merge");
    let union_branches = "SELECT key, payload FROM bench UNION ALL SELECT key, payload FROM bench";
    let cases = [
        ("SELECT key, payload FROM bench ORDER BY payload + 1".to_string(), None),
        (format!("SELECT key, payload FROM ({union_branches}) ORDER BY payload + 1"), None),
        (format!("SELECT key, payload FROM ({union_branches}) ORDER BY payload + 1 LIMIT 5"), None),
        (
            "SELECT a.key, a.payload FROM bench a JOIN (SELECT key FROM bench UNION ALL SELECT key FROM bench) b \
             ON a.key = b.key ORDER BY a.payload + 1"
                .to_string(),
            None,
        ),
        (
            "SELECT b.key, a.payload FROM (SELECT key FROM bench UNION ALL SELECT key FROM bench) b \
             JOIN bench a ON a.key = b.key ORDER BY a.payload + 1"
                .to_string(),
            None,
        ),
        (
            "SELECT key, payload FROM bench ORDER BY random()".to_string(),
            Some(("rewrite_expression", QueryGpuRemedyKind::UserAction)),
        ),
    ];
    for (sql, rejection) in cases {
        for target_partitions in [4, 1] {
            let plan = sql_plan(&dir, SessionConfig::new().with_target_partitions(target_partitions), &sql).await;
            let context = format!("target_partitions = {target_partitions}: {sql}");
            if target_partitions > 1 {
                assert!(plan_has_expression_keyed_merge(plan.as_ref()), "{context}");
            }
            let result = lower_physical_plan_to_native(plan, &ParquetReadStrategyConfig::default()).unwrap();
            match rejection {
                None => assert_eq!(result.selected_count(), 1, "{context}: {:?}", result.report.outcomes()),
                Some(expected) => {
                    assert_eq!(rejection_reason(&result), Some(CapabilityReason::UnsupportedExpression), "{context}");
                    let remedy = rejected_candidate_remedy(&result);
                    assert_eq!((remedy.code, remedy.kind), expected, "{context}");
                },
            }
        }
    }
    fs::remove_dir_all(dir).unwrap();
}

/// A merge's fetch and an outer offset apply to the one native sort by the
/// merge's expression key, so the native plan keeps the rows DataFusion's
/// merge returns.
#[tokio::test]
async fn expression_keyed_merge_fetch_limits_the_native_sort() {
    let dir = three_file_table("algeon-datafusion-expression-keyed-merge-fetch");
    for (sql, expected) in [
        ("SELECT key, payload FROM bench ORDER BY payload * 2 DESC LIMIT 2", LimitSpec::new(0, Some(2))),
        ("SELECT key, payload FROM bench ORDER BY payload * 2 DESC LIMIT 2 OFFSET 1", LimitSpec::new(1, Some(2))),
    ] {
        let plan = sql_plan(&dir, SessionConfig::new().with_target_partitions(4), sql).await;
        assert!(plan_has_expression_keyed_merge(plan.as_ref()), "{sql}");

        let result = lower_physical_plan_to_native(plan, &ParquetReadStrategyConfig::default()).unwrap();

        assert_eq!(result.selected_count(), 1, "{sql}: {:?}", result.report.outcomes());
        let native_plan = result.selected_plan();
        let limits = native_plan
            .nodes()
            .iter()
            .filter_map(|node| match &node.kind {
                PlanNode::Limit(limit) => Some(limit),
                _ => None,
            })
            .collect::<Vec<_>>();
        let [limit] = limits.as_slice() else {
            panic!("{sql}: expected one native limit:\n{native_plan}");
        };
        assert_eq!(limit.spec, expected, "{sql}");
        let PlanNode::Sort(sort) = &native_plan.node(limit.input).expect("limit input exists").kind else {
            panic!("{sql}: the limit reads the sort:\n{native_plan}");
        };
        assert!(matches!(sort.spec.expressions.as_slice(), [key] if key.descending), "{sql}: {:?}", sort.spec);
    }
    fs::remove_dir_all(dir).unwrap();
}

/// A per-partition sort that keeps fewer rows than the merge above it returns
/// has no single native sort, so the merge is rejected. Over one scan, one
/// partition leaves a single sort, and the remedy names the setting. Over
/// `UNION ALL` branches, which stay partitions at one partition, the merge and
/// its rejection remain, so the remedy is the engine gap.
#[tokio::test]
async fn merge_over_a_smaller_per_partition_fetch_names_one_partition_unless_it_reads_union_branches() {
    let dir = three_file_table("algeon-datafusion-merge-partition-fetch-remedy");
    let scan = sql_plan(&dir, SessionConfig::new().with_target_partitions(4), "SELECT key, payload FROM bench").await;
    assert!(scan.properties().output_partitioning().partition_count() > 1);
    let union: Arc<dyn ExecutionPlan> = UnionExec::try_new(vec![Arc::clone(&scan), Arc::clone(&scan)]).unwrap();
    let setting = ("set_target_partitions_one", QueryGpuRemedyKind::ConfigAction);
    let gap = ("unsupported_operator_gap", QueryGpuRemedyKind::EngineGap);
    for (input, expected) in [(scan, setting), (union, gap)] {
        let ordering = LexOrdering::new([PhysicalSortExpr::new(
            col("payload", &input.schema()).unwrap(),
            ArrowSortOptions::new(false, false),
        )])
        .unwrap();
        let sort: Arc<dyn ExecutionPlan> =
            Arc::new(SortExec::new(ordering.clone(), input).with_preserve_partitioning(true).with_fetch(Some(1)));
        let plan: Arc<dyn ExecutionPlan> = Arc::new(SortPreservingMergeExec::new(ordering, sort));

        let result = lower_physical_plan_to_native(plan, &ParquetReadStrategyConfig::default()).unwrap();

        assert_eq!(rejection_reason(&result), Some(CapabilityReason::NativeSortPreservingMergeUnsupported));
        let remedy = rejected_candidate_remedy(&result);
        assert_eq!((remedy.code, remedy.kind), expected);
    }
    fs::remove_dir_all(dir).unwrap();
}

/// The exact statistical mode refuses an aggregate DataFusion reads through
/// several partitions. The refusal names the aggregate function, and its
/// remedy names the setting that lifts the requirement.
#[test]
fn exact_statistical_mode_names_the_aggregate_over_partitioned_input() {
    let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Float64, false)]));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Float64Array::from(vec![1.0, 2.0]))]).unwrap();
    let stddev_over = |partitions: Vec<Vec<RecordBatch>>| -> Arc<dyn ExecutionPlan> {
        let source = MemorySourceConfig::try_new(&partitions, Arc::clone(&schema), None).unwrap();
        let scan = DataSourceExec::from_data_source(source) as Arc<dyn ExecutionPlan>;
        let stddev = AggregateExprBuilder::new(stddev_udaf(), vec![col("x", &schema).unwrap()])
            .schema(Arc::clone(&schema))
            .alias("spread")
            .build()
            .unwrap();
        Arc::new(
            AggregateExec::try_new(
                AggregateMode::Single,
                PhysicalGroupBy::new_single(Vec::new()),
                vec![Arc::new(stddev)],
                vec![None],
                scan,
                Arc::clone(&schema),
            )
            .unwrap(),
        )
    };

    let partitioned = stddev_over(vec![vec![batch.clone()], vec![batch.clone()]]);
    assert_eq!(
        floating_statistical_aggregate_input_is_partitioned(partitioned.as_ref()),
        Some(PlanBlocker::Aggregate { function: "stddev".to_string(), defect: AggregateDefect::PartitionedInput })
    );
    let single = stddev_over(vec![vec![batch]]);
    assert_eq!(floating_statistical_aggregate_input_is_partitioned(single.as_ref()), None);

    let remedy = CapabilityReason::NativeExactStatisticalInputPartitioned.facts().remedy;
    assert_eq!(remedy.kind, QueryGpuRemedyKind::ConfigAction);
    assert!(remedy.guidance.contains("gpu_native_tolerant"), "{}", remedy.guidance);
}
