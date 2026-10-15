use std::path::Path;

use datafusion::prelude::{ParquetReadOptions, SessionConfig};
use futures::executor::block_on;
use query_engine::rapids_backend::rapids_analysis_context;
use query_planning::{pipeline::CompiledPipeline, policy::PlanningPolicy};

use super::*;

#[tokio::test]
async fn native_optimizer_rule_rewrites_and_executes_bounded_join_aggregate_pipeline() {
    let dir = unique_temp_dir("algeon-datafusion-native-exec-bounded-join-aggregate");
    let left = dir.join("left.parquet");
    let right = dir.join("right.parquet");
    write_parquet_file(&left);
    write_parquet_file(&right);
    let plan = join_aggregate_sort_limit_sql_plan(&left, &right);
    let fixture = backend_fixture();
    let rule = AlgeonNativePhysicalOptimizerRule::new(template_optimizer_config());

    let (optimized, diagnostics) = rule.optimize_with_diagnostics(plan, &ConfigOptions::default()).unwrap();

    let native = optimized
        .downcast_ref::<AlgeonNativeExec>()
        .expect("single-join post-root pipeline should optimize to AlgeonNativeExec");
    let pipeline = compile_pipeline(
        native.native_plan(),
        &native.native_plan().capability_analysis(rapids_analysis_context(PlanningPolicy::default())),
    )
    .expect("lowered single-join post-root pipeline should compile");
    assert_eq!(task_kind_count(&pipeline, PipelineTaskKind::Aggregate), 1);
    assert_eq!(task_kind_count(&pipeline, PipelineTaskKind::FinalSort), 1);
    assert_eq!(task_kind_count(&pipeline, PipelineTaskKind::Limit), 1);
    let report = diagnostics.expect("native optimizer should record planning diagnostics").report;
    assert_eq!(report.selected_count(), 1);
    assert_eq!(report.not_selected_count(), 0);
    assert!(matches!(report.outcomes()[0], CandidateOutcome::Selected { .. }));

    let admitted = admitted_plan(&fixture, Arc::clone(&optimized));
    let batches = collect(admitted, Arc::new(TaskContext::default())).await.unwrap();
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 2);
    assert_eq!(batches[0].num_columns(), 2);
    assert_eq!(int32_array(&batches[0], 0).values().as_ref(), &[1, 2]);
    assert_eq!(int64_array(&batches[0], 1).values().as_ref(), &[303, 202]);

    let metrics = optimized.metrics().expect("native execution should expose DataFusion metrics");
    assert_eq!(metrics.sum_by_name("compiled_pipeline_attempts").unwrap().as_usize(), 1);
    // The native plan executes to a deferred host sink; the adapter's host
    // materialization counts as the pipeline's sink task.
    assert_eq!(metrics.sum_by_name("compiled_pipeline_tasks_executed").unwrap().as_usize(), pipeline.tasks().len() + 1);
    assert_eq!(metrics.sum_by_name("compiled_pipeline_host_arrow_sinks").unwrap().as_usize(), 1);

    fs::remove_dir_all(dir).unwrap();
}

fn task_kind_count(pipeline: &CompiledPipeline, kind: PipelineTaskKind) -> usize {
    pipeline.tasks().iter().filter(|task| task.kind() == kind).count()
}

fn join_aggregate_sort_limit_sql_plan(left: &Path, right: &Path) -> Arc<dyn ExecutionPlan> {
    let ctx = SessionContext::new_with_config(
        SessionConfig::new()
            .with_repartition_file_scans(false)
            .with_repartition_joins(false)
            .with_repartition_aggregations(false)
            .with_repartition_sorts(false)
            .with_target_partitions(1),
    );
    block_on(ctx.register_parquet("left_input", left.to_str().unwrap(), ParquetReadOptions::default())).unwrap();
    block_on(ctx.register_parquet("right_input", right.to_str().unwrap(), ParquetReadOptions::default())).unwrap();
    let sql = "
SELECT l.key, SUM(l.m00 + r.m01) AS sum_x00
FROM left_input l
JOIN right_input r ON l.key = r.key
WHERE l.payload >= 10 AND r.payload >= 10
GROUP BY l.key
ORDER BY sum_x00 DESC, l.key ASC
LIMIT 2
";
    let df = block_on(ctx.sql(sql)).unwrap();
    block_on(df.create_physical_plan()).unwrap()
}
