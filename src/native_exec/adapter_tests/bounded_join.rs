use super::*;

#[tokio::test]
async fn native_optimizer_rule_rewrites_and_executes_bounded_join_aggregate_pipeline() {
    let dir = unique_temp_dir("datafusion-nexus-native-exec-bounded-join-aggregate");
    let left = dir.join("left.parquet");
    let right = dir.join("right.parquet");
    write_parquet_file(&left);
    write_parquet_file(&right);
    let plan = join_aggregate_sort_limit_sql_plan(&left, &right);
    let fixture = backend_fixture();
    let rule = NexusNativePhysicalOptimizerRule::new(template_optimizer_config());

    let optimized = rule.optimize(plan, &ConfigOptions::default()).unwrap();

    let native = optimized
        .downcast_ref::<NexusNativeExec>()
        .expect("single-join post-root pipeline should optimize to NexusNativeExec");
    let pipeline = compile_pipeline(
        native.native_plan(),
        &native.native_plan().capability_analysis(nexus_query_engine::rapids_backend::rapids_analysis_context()),
    )
    .expect("lowered single-join post-root pipeline should compile");
    assert_eq!(task_kind_count(&pipeline, PipelineTaskKind::Aggregate), 1);
    assert_eq!(task_kind_count(&pipeline, PipelineTaskKind::FinalSort), 1);
    assert_eq!(task_kind_count(&pipeline, PipelineTaskKind::Limit), 1);
    let report = rule.last_report().expect("native optimizer should record a planning report");
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
    assert_eq!(metrics.sum_by_name("compiled_pipeline_tasks_executed").unwrap().as_usize(), pipeline.tasks().len());
    assert_eq!(metrics.sum_by_name("compiled_pipeline_host_arrow_sinks").unwrap().as_usize(), 1);

    fs::remove_dir_all(dir).unwrap();
}

fn task_kind_count(pipeline: &nexus_query_engine::pipeline::CompiledPipeline, kind: PipelineTaskKind) -> usize {
    pipeline.tasks().iter().filter(|task| task.kind() == kind).count()
}
#[tokio::test]
async fn native_exec_preserves_bounded_join_rejection_facts_from_native_runtime() {
    let dir = unique_temp_dir("datafusion-nexus-native-bounded-join-runtime-rejection");
    let left = dir.join("left.parquet");
    let right = dir.join("right.parquet");
    write_join_left_parquet_file(&left);
    write_join_right_parquet_file(&right);
    let native_plan = bounded_left_build_left_semi_join_host_sink_plan_with_paths(&left, &right);
    let fixture = backend_fixture();
    let config = bound_execution_config(&fixture);
    let exec = Arc::new(NexusNativeExec::try_new(native_plan, config).unwrap()) as Arc<dyn ExecutionPlan>;

    let error = collect(exec, Arc::new(TaskContext::default())).await.unwrap_err();
    let adapter_error = adapter_error(&error);

    assert_eq!(adapter_error.code(), ErrorCode::UnsupportedOperator);
    assert_eq!(adapter_error.fact_value("native_code"), Some("unsupported_plan_shape"));
    assert_eq!(adapter_error.fact_value("native_operation"), Some("nexus_query_engine.bounded.plan"));
    assert_eq!(adapter_error.fact_value("reason"), Some("bounded_join_plan_shape_unsupported"));
    assert_eq!(adapter_error.fact_value("bounded_join_rejection_scope"), Some("node"));
    assert_eq!(adapter_error.fact_value("bounded_join_rejection_category"), Some("unsupported_node_operator"));
    assert_eq!(adapter_error.fact_value("bounded_join_rejection_reason"), Some("unsupported_join_node_capability"));
    assert_eq!(adapter_error.fact_value("path"), None);
    assert_eq!(adapter_error.fact_value("file_index"), None);
    assert_eq!(adapter_error.fact_value("row_group_indexes"), None);
    assert_eq!(adapter_error.fact_value("row_group_count"), None);
    assert_eq!(adapter_error.fact_value("chunk_index"), None);

    fs::remove_dir_all(dir).unwrap();
}
#[tokio::test]
async fn native_composed_exec_preserves_capability_facts_on_runtime_rejection() {
    let dir = unique_temp_dir("datafusion-nexus-native-composed-bounded-join-rejection");
    let left = dir.join("left.parquet");
    let right = dir.join("right.parquet");
    write_parquet_file(&left);
    write_parquet_file(&right);
    let input_schema = parquet_file_schema();

    let left_producer = Arc::new(
        NexusNativeGpuProducerExec::try_new(
            "left_input",
            local_parquet_gpu_resident_plan(&left, Arc::clone(&input_schema)),
        )
        .unwrap(),
    ) as Arc<dyn ExecutionPlan>;
    let right_producer = Arc::new(
        NexusNativeGpuProducerExec::try_new(
            "right_input",
            local_parquet_gpu_resident_plan(&right, Arc::clone(&input_schema)),
        )
        .unwrap(),
    ) as Arc<dyn ExecutionPlan>;
    let consumer_plan =
        bounded_gpu_resident_left_build_left_semi_join_host_sink_plan("left_input", "right_input", input_schema);
    let fixture = backend_fixture();
    let exec = Arc::new(
        NexusNativeComposedExec::try_new(
            consumer_plan,
            vec![left_producer, right_producer],
            bound_execution_config(&fixture),
        )
        .unwrap(),
    ) as Arc<dyn ExecutionPlan>;

    let error = collect(exec, Arc::new(TaskContext::default())).await.unwrap_err();
    let adapter_error = adapter_error(&error);

    assert_eq!(adapter_error.code(), ErrorCode::UnsupportedOperator);
    assert_eq!(adapter_error.fact_value("reason"), Some("bounded_join_plan_shape_unsupported"));
    assert!(adapter_error.fact_value("bounded_join_rejection_scope").is_some());
    assert!(adapter_error.fact_value("bounded_join_rejection_owner").is_some());
    assert!(adapter_error.fact_value("bounded_join_rejection_reason").is_some());
    assert_eq!(adapter_error.fact_value("modeled_reservations"), None);
    assert_eq!(adapter_error.fact_value("bounded_join_reservation_requirements"), None);

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_exec_reports_capability_error_facts_for_unsupported_bounded_join_runtime_shape() {
    let dir = unique_temp_dir("datafusion-nexus-native-bounded-join-runtime-shape");
    let left = dir.join("left.parquet");
    let right = dir.join("right.parquet");
    write_join_left_parquet_file(&left);
    write_join_right_parquet_file(&right);
    let native_plan = bounded_left_build_left_semi_join_host_sink_plan_with_paths(&left, &right);
    let fixture = backend_fixture();
    let exec = Arc::new(NexusNativeExec::try_new(native_plan, bound_execution_config(&fixture)).unwrap())
        as Arc<dyn ExecutionPlan>;
    let error = collect(exec, Arc::new(TaskContext::default())).await.unwrap_err();
    let runtime_error = adapter_error(&error);

    assert_eq!(runtime_error.fact_value("reason"), Some("bounded_join_plan_shape_unsupported"));
    assert!(runtime_error.fact_value("bounded_join_rejection_scope").is_some());
    assert!(runtime_error.fact_value("bounded_join_rejection_boundary").is_some());
    assert!(runtime_error.fact_value("bounded_join_rejection_category").is_some());
    assert_eq!(runtime_error.fact_value("bounded_join_reservation_model"), None);
    assert_eq!(runtime_error.fact_value("modeled_reservations"), None);

    fs::remove_dir_all(dir).unwrap();
}
