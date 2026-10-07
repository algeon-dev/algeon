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
        ("SELECT key, payload FROM bench ORDER BY payload + 1", CapabilityReason::NativeSortPreservingMergeUnsupported),
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

/// A merge of `UNION ALL` branches stays at one partition, and a sort key
/// without a native form is rejected by the single sort DataFusion plans there.
/// Both carry an engine gap at either partition count, and the single sort
/// reports the expression it cannot lower. The last query unions on the
/// collected side of a join, which DataFusion merges into one partition, so the
/// setting still removes the merge.
#[tokio::test]
async fn sort_preserving_merge_remedy_is_an_engine_gap_when_one_partition_cannot_lower_it() {
    let dir = three_file_table("algeon-datafusion-merge-gap-remedy");
    let union_branches = "SELECT key, payload FROM bench UNION ALL SELECT key, payload FROM bench";
    let gap = ("unsupported_operator_gap", QueryGpuRemedyKind::EngineGap);
    let setting = ("set_target_partitions_one", QueryGpuRemedyKind::ConfigAction);
    let cases = [
        (
            format!("SELECT key, payload FROM ({union_branches}) ORDER BY payload + 1"),
            gap,
            Some((CapabilityReason::NativeSortPreservingMergeUnsupported, gap)),
        ),
        (
            format!("SELECT key, payload FROM ({union_branches}) ORDER BY payload + 1 LIMIT 5"),
            gap,
            Some((CapabilityReason::NativeSortPreservingMergeUnsupported, gap)),
        ),
        (
            "SELECT a.key, a.payload FROM bench a JOIN (SELECT key FROM bench UNION ALL SELECT key FROM bench) b \
             ON a.key = b.key ORDER BY a.payload + 1"
                .to_string(),
            gap,
            Some((CapabilityReason::NativeSortPreservingMergeUnsupported, gap)),
        ),
        (
            "SELECT key, payload FROM bench ORDER BY random()".to_string(),
            gap,
            Some((CapabilityReason::UnsupportedExpression, ("rewrite_expression", QueryGpuRemedyKind::UserAction))),
        ),
        (
            "SELECT b.key, a.payload FROM (SELECT key FROM bench UNION ALL SELECT key FROM bench) b \
             JOIN bench a ON a.key = b.key ORDER BY a.payload + 1"
                .to_string(),
            setting,
            None,
        ),
    ];
    for (sql, remedy_at_four, at_one) in cases {
        let plan = sql_plan(&dir, SessionConfig::new().with_target_partitions(4), &sql).await;
        let result = lower_physical_plan_to_native(plan, &ParquetReadStrategyConfig::default()).unwrap();
        let context = format!("target_partitions = 4: {sql}");
        assert_eq!(
            rejection_reason(&result),
            Some(CapabilityReason::NativeSortPreservingMergeUnsupported),
            "{context}"
        );
        let remedy = rejected_candidate_remedy(&result);
        assert_eq!((remedy.code, remedy.kind), remedy_at_four, "{context}");

        let plan = sql_plan(&dir, SessionConfig::new().with_target_partitions(1), &sql).await;
        let result = lower_physical_plan_to_native(plan, &ParquetReadStrategyConfig::default()).unwrap();
        let context = format!("target_partitions = 1: {sql}");
        match at_one {
            Some((reason, expected)) => {
                assert_eq!(rejection_reason(&result), Some(reason), "{context}");
                let remedy = rejected_candidate_remedy(&result);
                assert_eq!((remedy.code, remedy.kind), expected, "{context}");
            },
            None => assert_eq!(result.selected_count(), 1, "{context}: {:?}", result.report.outcomes()),
        }
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
