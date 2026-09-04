use nexus_query_engine::plan::{AdvisoryStatistic, AdvisoryStatisticsScope};

use super::*;

#[tokio::test]
async fn shadow_advisory_binds_projection_and_final_aggregate_outputs() {
    let dir = unique_temp_dir("datafusion-nexus-native-lowering-shadow-advisory");
    let path = dir.join("input.parquet");
    write_numeric_parquet_file(&path);
    let plan = sql_plan(
        &path,
        direct_friendly_sql_session_config(),
        "SELECT key, SUM(payload) AS total FROM bench WHERE payload > 1 GROUP BY key",
    )
    .await;
    let lowered = lower_physical_plan_to_native(plan);
    assert_eq!(lowered.selected_count(), 1, "{:?}", lowered.report.outcomes());
    let native = &lowered.plans[0];
    let aggregate = native.nodes().iter().find(|node| matches!(node.kind, PlanNode::Aggregate(_))).unwrap();
    let binding = native
        .advisory_outputs()
        .iter()
        .find(|binding| binding.node == aggregate.id)
        .expect("final aggregate output has frontend statistics even when unknown");
    assert_eq!(&binding.schema, aggregate.output_schema());
    assert_eq!(binding.scope, AdvisoryStatisticsScope::WholeOutput);
    assert_eq!(binding.frontend_version, datafusion::DATAFUSION_VERSION);

    let projected =
        sql_plan(&path, direct_friendly_sql_session_config(), "SELECT payload + 1 AS value FROM bench").await;
    let lowered = lower_physical_plan_to_native(projected);
    assert_eq!(lowered.selected_count(), 1, "{:?}", lowered.report.outcomes());
    let native = &lowered.plans[0];
    let binding = native
        .advisory_outputs()
        .iter()
        .find(|binding| binding.node == native.root_id())
        .expect("root projection carries its own frontend estimate");
    assert!(!matches!(binding.statistics.rows, AdvisoryStatistic::Absent));
    assert_eq!(&binding.schema, native.root_node().output_schema());
    fs::remove_dir_all(dir).unwrap();
}
