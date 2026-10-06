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
/// and the config remedy of its reason names the session setting that avoids it.
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
            "SELECT key, COUNT(*) FROM bench GROUP BY key LIMIT 2",
            CapabilityReason::NativeLocalLimitPartitionUnsupported,
        ),
    ];
    for (sql, reason) in cases {
        let partitioned = sql_plan(&dir, SessionConfig::new().with_target_partitions(4), sql).await;
        let result = lower_physical_plan_to_native(partitioned, &ParquetReadStrategyConfig::default()).unwrap();
        assert_eq!(rejection_reason(&result), Some(reason), "{sql}");

        let single = sql_plan(&dir, SessionConfig::new().with_target_partitions(1), sql).await;
        let result = lower_physical_plan_to_native(single, &ParquetReadStrategyConfig::default()).unwrap();
        assert_eq!(result.selected_count(), 1, "{sql}: {:?}", result.report.outcomes());

        let remedy = reason.facts().remedy;
        assert_eq!(remedy.kind, QueryGpuRemedyKind::ConfigAction, "{reason:?}");
        assert!(remedy.guidance.contains("SET datafusion.execution.target_partitions = 1"), "{reason:?}");
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
