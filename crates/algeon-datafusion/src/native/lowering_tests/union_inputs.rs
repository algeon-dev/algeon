use super::*;

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
