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
    let pipeline = compile_pipeline(native.native_plan(), &native.native_plan().capability_analysis())
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
    assert_eq!(adapter_error.fact_value("bounded_join_enabled"), Some("true"));
    assert_eq!(
        adapter_error.fact_value("boundary_policy"),
        Some(
            "source_read=direct_device_local_parquet,build_materialization=query_scoped_gpu_hash_join,probe_execution=reusable_gpu_hash_join,result_materialization=explicit_sink"
        )
    );
    assert_eq!(adapter_error.fact_value("source_read_boundary"), Some("direct_device_local_parquet"));
    assert_eq!(adapter_error.fact_value("build_materialization_boundary"), Some("query_scoped_gpu_hash_join"));
    assert_eq!(adapter_error.fact_value("probe_execution_boundary"), Some("reusable_gpu_hash_join"));
    assert_eq!(adapter_error.fact_value("result_materialization_boundary"), Some("explicit_sink"));
    assert_eq!(adapter_error.fact_value("blocked_reservations"), Some(""));
    assert_eq!(adapter_error.fact_value("bounded_join_rejection_scope"), Some("node"));
    assert_eq!(adapter_error.fact_value("bounded_join_rejection_category"), Some("unsupported_node_operator"));
    assert_eq!(adapter_error.fact_value("bounded_join_rejection_reason"), Some("unsupported_join_node_capability"));
    let modeled = adapter_error.fact_value("modeled_reservations").expect("modeled reservations fact");
    for expected in ["source_read", "build_side", "probe_side", "gather_map", "join_output", "materialization"] {
        assert!(modeled.split(',').any(|kind| kind == expected), "missing modeled reservation {expected}");
    }
    assert_eq!(adapter_error.fact_value("bounded_join_lower_level_blockers"), Some(""));
    assert_eq!(
        adapter_error.fact_value("cudf_rs_dataframe_gather_output_reservation_contract"),
        Some("dataframe_gather_output_reservation_size_bytes")
    );
    assert_eq!(
        adapter_error.fact_value("cudf_rs_filtered_join_hash_table_observability"),
        Some("filtered_join_device_allocated_size_bytes")
    );
    assert_eq!(
        adapter_error.fact_value("cudf_rs_filtered_join_output_size_observability"),
        Some("filtered_join_probe_state_output_size")
    );
    assert_eq!(
        adapter_error.fact_value("cudf_rs_filtered_join_probe_state_contract"),
        Some("filtered_join_retained_probe_state")
    );
    assert_eq!(
        adapter_error.fact_value("cudf_rs_filtered_join_probe_state_observability"),
        Some("filtered_join_probe_device_allocated_size_bytes")
    );
    assert_eq!(
        adapter_error.fact_value("cudf_rs_filtered_join_probe_materialization_contract"),
        Some("filtered_join_probe_state_materialize_indices")
    );
    assert_eq!(
        adapter_error.fact_value("cudf_rs_conditional_join_output_size_observability"),
        Some("conditional_join_exact_output_size")
    );
    assert_eq!(
        adapter_error.fact_value("cudf_rs_conditional_join_output_size_hint"),
        Some("conditional_join_indices_with_output_size")
    );
    assert_eq!(adapter_error.fact_value("bounded_join_execution_family_alias_compatibility"), None);
    assert_eq!(
        adapter_error.fact_value("cudf_rs_mixed_join_size_data_alias_compatibility"),
        Some(
            "mixed_inner_left_join_size_data_values_are_legacy_aliases_for_mixed_keyed_residual_inner_left_full_left_semi_left_anti_size_data"
        )
    );
    assert_eq!(
        adapter_error.fact_value("bounded_join_mixed_keyed_residual_capability_detail"),
        Some("mixed_keyed_residual_root_join_types=inner,left,full,left_semi,left_anti")
    );
    assert_eq!(adapter_error.fact_value("path"), None);
    assert_eq!(adapter_error.fact_value("file_index"), None);
    assert_eq!(adapter_error.fact_value("row_group_indexes"), None);
    assert_eq!(adapter_error.fact_value("row_group_count"), None);
    assert_eq!(adapter_error.fact_value("chunk_index"), None);

    fs::remove_dir_all(dir).unwrap();
}
#[tokio::test]
async fn native_composed_exec_preserves_deferred_host_materialization_facts_on_runtime_rejection() {
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
    assert_eq!(adapter_error.fact_value("result_materialization_boundary"), Some("explicit_sink"));
    assert!(
        adapter_error
            .fact_value("modeled_reservations")
            .is_some_and(|value| value.split(',').any(|kind| kind == "materialization")),
        "composed runtime rejection must retain the caller-visible host materialization reservation: {adapter_error}"
    );
    assert!(
        adapter_error
            .fact_value("bounded_join_reservation_statuses")
            .is_some_and(|value| value.split(',').any(|status| status == "materialization:modeled")),
        "composed runtime rejection must retain the modeled materialization status: {adapter_error}"
    );
    assert!(
        adapter_error.fact_value("bounded_join_reservation_requirements").is_some_and(|value| value.contains(
            "materialization:query_engine_explicit_sink_materialization_reservation:query_engine_owned_boundary"
        )),
        "composed runtime rejection must retain the explicit sink materialization requirement: {adapter_error}"
    );

    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn native_exec_reports_legacy_error_facts_for_unsupported_bounded_join_runtime_shape() {
    let dir = unique_temp_dir("datafusion-nexus-native-bounded-join-runtime-shape");
    let left = dir.join("left.parquet");
    let right = dir.join("right.parquet");
    write_join_left_parquet_file(&left);
    write_join_right_parquet_file(&right);
    let expected_unsupported_shape = BoundedJoinReservationContract.supported_shape_requirement();

    let native_plan = bounded_left_build_left_semi_join_host_sink_plan_with_paths(&left, &right);
    let fixture = backend_fixture();
    let exec = Arc::new(NexusNativeExec::try_new(native_plan, bound_execution_config(&fixture)).unwrap())
        as Arc<dyn ExecutionPlan>;
    let error = collect(exec, Arc::new(TaskContext::default())).await.unwrap_err();
    let runtime_error = adapter_error(&error);

    assert_eq!(runtime_error.fact_value("bounded_join_enabled"), Some("true"));
    assert_eq!(runtime_error.fact_value("bounded_join_execution_status"), Some("unsupported_plan_shape"));
    assert_eq!(runtime_error.fact_value("bounded_join_unsupported_shape"), Some(expected_unsupported_shape.as_str()));
    assert_eq!(runtime_error.fact_value("bounded_join_execution_families"), None);
    assert_eq!(runtime_error.fact_value("reason"), Some("bounded_join_plan_shape_unsupported"));
    let modeled = runtime_error.fact_value("modeled_reservations").expect("modeled reservations fact");
    for expected in ["source_read", "build_side", "probe_side", "gather_map", "join_output", "materialization"] {
        assert!(modeled.split(',').any(|kind| kind == expected), "missing modeled reservation {expected}");
    }
    assert_eq!(runtime_error.fact_value("blocked_reservations"), Some(""));
    assert_eq!(runtime_error.fact_value("bounded_join_execution_family_alias_compatibility"), None);
    assert_eq!(
        runtime_error.fact_value("cudf_rs_mixed_join_size_data_alias_compatibility"),
        Some(
            "mixed_inner_left_join_size_data_values_are_legacy_aliases_for_mixed_keyed_residual_inner_left_full_left_semi_left_anti_size_data"
        )
    );
    assert_eq!(
        runtime_error.fact_value("bounded_join_mixed_keyed_residual_capability_detail"),
        Some("mixed_keyed_residual_root_join_types=inner,left,full,left_semi,left_anti")
    );

    fs::remove_dir_all(dir).unwrap();
}
