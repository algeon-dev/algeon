use query_engine::{
    admission::bounded_pipeline_decision,
    mode::StatisticalAggregateExecutionMode,
    pipeline::ShadowBytes,
    plan::{AdvisoryStatistic, AdvisoryStatisticsScope},
    shadow::estimate_compiled_pipeline,
};

use super::*;

#[tokio::test]
async fn shadow_advisory_binds_projection_and_final_aggregate_outputs() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-shadow-advisory");
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

#[tokio::test]
async fn shadow_estimate_composes_frontend_advisory_with_the_compiled_schedule() {
    let dir = unique_temp_dir("algeon-datafusion-native-lowering-shadow-estimate");
    write_tiny_tpch_parquet_dataset(&dir).unwrap();
    let items = tpch_fallback_pressure_workload_items(&dir, tpch_report_session_config()).await.unwrap();
    let q3 = items.iter().find(|item| item.query_id == "q03").expect("TPC-H workload should include q03");
    let NativeFallbackPressureWorkloadInput::PhysicalPlan(plan) = &q3.input else {
        panic!("q03 should produce a physical plan");
    };
    let lowered = lower_physical_plan_to_native(Arc::clone(plan));
    assert_eq!(lowered.selected_count(), 1, "{:?}", lowered.report.outcomes());
    assert!(!lowered.plans[0].advisory_outputs().is_empty(), "q03 lowering attaches frontend statistics");

    // The estimate is composed over the schedule that actually executes, so the
    // advisory volumes must survive the optimizer rewrites that produce it.
    let native = &query_engine::optimizer::optimize_query_plan(
        lowered.plans[0].clone(),
        &query_engine::optimizer::NativeOptimizerOptions::default(),
        query_engine::rapids_backend::rapids_analysis_context(),
        query_engine::dimension_pruning::local_dimension_pruning_resolver(),
    )
    .plan;
    assert!(!native.advisory_outputs().is_empty(), "optimizer rewrites carry frontend statistics forward");
    let admission = bounded_pipeline_decision(
        native,
        StatisticalAggregateExecutionMode::default(),
        query_engine::rapids_backend::rapids_analysis_context(),
    )
    .unwrap()
    .into_execution_admission()
    .unwrap()
    .expect("the lowered q03 join plan compiles to a bounded pipeline");
    let pipeline = admission.pipeline();

    let estimate = estimate_compiled_pipeline(native, pipeline).unwrap();
    assert_eq!(estimate.pipeline.envelopes.len(), pipeline.tasks().len());
    assert_eq!(estimate.pipeline.task_workspace.len(), pipeline.tasks().len());
    assert_eq!(estimate.pipeline.resources.len(), pipeline.resources().len());
    assert_eq!(estimate.frontend_outputs.len(), native.advisory_outputs().len());
    for (binding, captured) in native.advisory_outputs().iter().zip(&estimate.frontend_outputs) {
        assert_eq!(captured["node"].as_u64(), Some(binding.node.index() as u64));
        assert_eq!(captured["frontend"].as_str(), Some(binding.frontend.as_str()));
        assert_eq!(captured["frontend_version"].as_str(), Some(binding.frontend_version.as_str()));
    }

    // Local Parquet decode and every operator workspace still lack a component
    // allocation model, so the composed peak stays unknown with its reason
    // instead of reporting an invented zero.
    assert!(
        estimate.pipeline.resources.iter().any(|resource| matches!(
            &resource.model.bytes,
            ShadowBytes::Unknown { reason } if reason.contains("decode")
        )),
        "{:?}",
        estimate.pipeline.resources
    );
    assert!(matches!(estimate.pipeline.query_owned_peak, ShadowBytes::Unknown { .. }));
    assert!(matches!(estimate.pipeline.borrowed_resident_peak, ShadowBytes::Estimated { bytes: 0, .. }));

    fs::remove_dir_all(dir).unwrap();
}
