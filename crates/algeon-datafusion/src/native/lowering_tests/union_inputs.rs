use datafusion::physical_plan::displayable;

use super::*;
use crate::native::ParquetReadStrategyConfig;

/// The plan builder interns identical nodes, so UNION ALL branches that read
/// the same table, or the same aggregate CTE, become one native node that the
/// union reads through two inputs. The rule keeps such a plan native.
#[tokio::test]
async fn unions_reading_one_native_node_twice_plan_natively() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-union-inputs");
    write_numeric_parquet_file(&dir.join("part0.parquet"));

    for sql in [
        "SELECT key FROM bench UNION ALL SELECT key FROM bench",
        // DataFusion prunes the unused window, which leaves the same scan twice.
        "SELECT key FROM bench UNION ALL \
         SELECT key FROM (SELECT key, ROW_NUMBER() OVER (ORDER BY payload, key) AS rk FROM bench) x",
        "WITH c AS (SELECT key, SUM(m00) AS s FROM bench GROUP BY key) \
         SELECT key AS x, s FROM c UNION ALL SELECT key, s FROM c",
    ] {
        let plan = sql_plan(&dir, direct_friendly_sql_session_config(), sql).await;

        let (report, native_plan) = optimize_with_native_rule(plan);

        assert!(
            matches!(report.outcomes(), [CandidateOutcome::Selected { .. }]),
            "{sql}\noutcomes={:?}",
            report.outcomes()
        );
        let native_plan = native_plan.expect("the rule keeps the native plan");
        assert!(
            native_plan.nodes().iter().any(|node| {
                matches!(node.kind, PlanNode::Union(_))
                    && matches!(node.input_ids(), [first, second] if first == second)
            }),
            "{sql}\nthe union reads one node through both inputs:\n{native_plan}"
        );
    }

    fs::remove_dir_all(dir).unwrap();
}

/// DataFusion plans a `UNION ALL` whose branches share one hash partitioning
/// as an `InterleaveExec`, which emits every branch row once like a union.
/// The candidate keeps the operator's name and lowers to one native union.
#[tokio::test]
async fn interleaved_union_of_partitioned_joins_lowers_to_a_native_union() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-interleave");
    for index in 0..3 {
        write_numeric_parquet_file(&dir.join(format!("part{index}.parquet")));
    }
    let mut config = SessionConfig::new().with_target_partitions(4);
    config.options_mut().optimizer.hash_join_single_partition_threshold = 0;
    config.options_mut().optimizer.hash_join_single_partition_threshold_rows = 0;
    let sql = "SELECT key, COUNT(*) AS n FROM (\
               SELECT a.key FROM bench a JOIN bench b ON a.key = b.key WHERE a.payload >= 10 \
               UNION ALL SELECT a.key FROM bench a JOIN bench b ON a.key = b.key WHERE a.payload < 10) \
               GROUP BY key";
    let plan = sql_plan(&dir, config, sql).await;
    let tree = displayable(plan.as_ref()).indent(false).to_string();
    assert!(tree.contains("InterleaveExec"), "{tree}");

    let result = lower_physical_plan_to_native(plan, &ParquetReadStrategyConfig::default()).unwrap();

    let [CandidateOutcome::Selected { candidate }] = result.report.outcomes() else {
        panic!("{tree}\noutcomes={:?}", result.report.outcomes());
    };
    assert!(candidate.shape.contains_operator("InterleaveExec"), "{}", candidate.shape);
    let unions = result.selected_plan().nodes().iter().filter(|node| matches!(node.kind, PlanNode::Union(_))).count();
    assert_eq!(unions, 1, "{}", result.selected_plan());

    fs::remove_dir_all(dir).unwrap();
}
