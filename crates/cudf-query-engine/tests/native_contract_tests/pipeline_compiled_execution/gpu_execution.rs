use super::*;

#[test]
fn compiled_gpu_executor_runs_root_hash_join_build_probe_materialize() {
    let Ok(runtime) = cudf::Runtime::new() else {
        return;
    };
    let dir = unique_temp_dir("cudf-query-engine-compiled-d14");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    write_join_parquets(&left_path, &right_path);

    let compiled_plan = root_join_data_plan(&left_path, &right_path);
    let legacy_plan = root_join_data_plan(&left_path, &right_path);
    let analysis = compiled_plan.capability_analysis();
    let pipeline = compile_pipeline(&compiled_plan, &analysis).expect("root join should compile");
    assert_eq!(
        compiled_bounded_pipeline_support(&pipeline).expect("support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );
    let scheduler = CompiledPipelineScheduler::new(&pipeline).expect("pipeline should be schedulable");
    let context = bounded_context(Arc::new(runtime));
    let mut compiled_metrics = QueryMetrics::default();
    let gpu_inputs = cudf_query_engine::GpuResidentInputMap::new();

    let (state, compiled_batch) = {
        let mut executor =
            CompiledGpuTaskExecutor::new(&context, &compiled_plan, &pipeline, &gpu_inputs, &mut compiled_metrics)
                .unwrap();
        let state = scheduler.run(&mut executor).expect("compiled root hash join should execute");
        let output = executor.take_host_output().expect("compiled host sink should materialize");
        (state, output)
    };
    let production_result = execute_query_plan(&context, &legacy_plan).expect("bounded root join should execute");
    let production_batch = production_result.output().as_host_arrow().expect("production output should be host Arrow");

    assert_eq!(
        state.execution_order(),
        &[
            PipelineTaskId::new(0),
            PipelineTaskId::new(1),
            PipelineTaskId::new(2),
            PipelineTaskId::new(3),
            PipelineTaskId::new(4),
            PipelineTaskId::new(5),
        ]
    );
    assert_eq!(state.live_resource_count(), 0);
    assert_eq!(compiled_batch.num_rows(), production_batch.num_rows());
    assert_eq!(int32_values(&compiled_batch, 0), int32_values(production_batch, 0));
    assert_eq!(int32_values(&compiled_batch, 1), int32_values(production_batch, 1));
    assert_eq!(int32_values(&compiled_batch, 0), vec![Some(2), Some(4)]);
    assert_eq!(int32_values(&compiled_batch, 1), vec![Some(200), Some(400)]);
    assert_eq!(compiled_metrics.hash_join_build_count, 1);
    assert!(compiled_metrics.hash_join_hash_table_bytes > 0);
}

#[test]
fn compiled_gpu_executor_runs_shared_source_hash_join_without_rereading() {
    let Ok(runtime) = cudf::Runtime::new() else {
        return;
    };
    let dir = unique_temp_dir("cudf-query-engine-compiled-d32-shared-source");
    let path = dir.join("shared.parquet");
    write_self_join_parquet(&path);

    let plan = shared_source_self_join_data_plan(&path);
    let analysis = plan.capability_analysis();
    let pipeline = compile_pipeline(&plan, &analysis).expect("shared-source self join should compile");
    assert_task_kind_count(&pipeline, PipelineTaskKind::SourceRead, 1);
    let shared_resource = pipeline
        .resources()
        .iter()
        .find(|resource| resource.consumers().len() == 2)
        .expect("one source resource should feed both join inputs")
        .id();
    assert_eq!(
        compiled_bounded_query_plan_support(&plan, &pipeline).expect("plan-aware support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );
    let scheduler = CompiledPipelineScheduler::new(&pipeline).expect("pipeline should be schedulable");
    let context = bounded_context(Arc::new(runtime));
    let mut metrics = QueryMetrics::default();
    let gpu_inputs = cudf_query_engine::GpuResidentInputMap::new();

    let (state, compiled_batch) = {
        let mut executor = CompiledGpuTaskExecutor::new(&context, &plan, &pipeline, &gpu_inputs, &mut metrics).unwrap();
        let state = scheduler.run(&mut executor).expect("compiled shared-source join should execute");
        let output = executor.take_host_output().expect("compiled host sink should materialize");
        (state, output)
    };

    assert_eq!(
        state.execution_order(),
        &[
            PipelineTaskId::new(0),
            PipelineTaskId::new(1),
            PipelineTaskId::new(2),
            PipelineTaskId::new(3),
            PipelineTaskId::new(4),
        ]
    );
    assert!(state.is_resource_released(shared_resource));
    assert_eq!(state.live_resource_count(), 0);
    assert_eq!(metrics.source_chunks_committed, 1);
    assert_eq!(compiled_batch.num_rows(), 3);
    assert_eq!(int32_values(&compiled_batch, 0), vec![Some(1), Some(2), Some(4)]);
    assert_eq!(int32_values(&compiled_batch, 1), vec![Some(10), Some(20), Some(40)]);
}

#[test]
fn execute_query_plan_routes_grouped_aggregate_join_build_input_through_compiled_pipeline() {
    let Ok(runtime) = cudf::Runtime::new() else {
        return;
    };
    let dir = unique_temp_dir("cudf-query-engine-compiled-d33-aggregate-join-input");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    write_aggregate_join_input_parquets(&left_path, &right_path);

    let optimizer_options = NativeOptimizerOptions {
        join_build_side_selection: JoinBuildSideSelectionOptions::disabled(),
        ..NativeOptimizerOptions::default()
    };
    let plan =
        optimize_query_plan(aggregate_join_build_input_data_plan(&left_path, &right_path), &optimizer_options).plan;
    assert_eq!(plan.dynamic_filter_routes().len(), 1);
    assert!(matches!(plan.dynamic_filter_routes()[0].producer, DynamicFilterProducer::AggregateGroupKeys { .. }));
    let analysis = plan.capability_analysis();
    let pipeline = compile_pipeline(&plan, &analysis).expect("aggregate join input plan should compile");
    assert_eq!(
        compiled_bounded_query_plan_support(&plan, &pipeline).expect("plan-aware support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );
    let context = bounded_context(Arc::new(runtime));

    let result = execute_query_plan(&context, &plan)
        .expect("D.3.3 grouped aggregate join-build input should execute through compiled path");
    let batch = result.output().as_host_arrow().expect("compiled output should be host Arrow");
    let metrics = result.metrics();

    assert_eq!(batch.num_rows(), 2);
    assert_eq!(int32_int64_pairs_sorted(batch, 0, 1), vec![(Some(1), Some(15)), (Some(2), Some(20))]);
    assert_eq!(metrics.compiled_pipeline_attempts, 1);
    assert_eq!(metrics.compiled_pipeline_tasks_executed, pipeline.tasks().len() as u64);
    assert_eq!(metrics.compiled_pipeline_host_arrow_sinks, 1);
    assert_eq!(metrics.dynamic_filter_produced_count, 1);
    assert_eq!(metrics.dynamic_filter_applied_count, 1);
    assert_eq!(metrics.hash_join_build_count, 1);
    assert_eq!(metrics.one_shot_join_count, 0);
    let join_strategies = metrics
        .operators
        .iter()
        .filter_map(|operator| operator.join.as_ref())
        .map(|join| join.execution_strategy.as_str())
        .collect::<Vec<_>>();
    assert!(
        join_strategies.contains(&"bounded_reusable_hash_join_single_pass_probe"),
        "join strategies: {join_strategies:?}"
    );
}

#[test]
fn execute_query_plan_routes_d3_6_nonlinear_topk_and_subquery_filter_paths_through_compiled_pipeline() {
    let Ok(runtime) = cudf::Runtime::new() else {
        return;
    };
    let runtime = Arc::new(runtime);
    write_d3_6_contract_parquets();

    for (label, plan) in [
        ("aggregate-over-join-subplan TopK", aggregate_over_join_subplan_topk_plan()),
        ("resident grouped-aggregate TopK", resident_grouped_aggregate_input_post_join_aggregate_topk_plan()),
        ("resident grouped-aggregate subquery filter", resident_grouped_aggregate_subquery_filter_plan()),
    ] {
        let analysis = plan.capability_analysis();
        let pipeline =
            compile_pipeline(&plan, &analysis).unwrap_or_else(|error| panic!("{label}: compile failed: {error}"));
        assert_eq!(
            compiled_bounded_query_plan_support(&plan, &pipeline).expect("plan-aware support check should not fail"),
            CompiledBoundedPipelineSupport::Supported,
            "{label}: D.3.6 plan should be admitted by compiled support"
        );
        let context = bounded_context(Arc::clone(&runtime));

        let result = execute_query_plan(&context, &plan)
            .unwrap_or_else(|error| panic!("{label}: D.3.6 compiled execution failed: {error}"));
        let batch = result.output().as_host_arrow().expect("D.3.6 output should be host Arrow");
        let metrics = result.metrics();

        assert!(batch.num_rows() > 0, "{label}: compiled execution should produce rows");
        assert_eq!(metrics.compiled_pipeline_attempts, 1, "{label}: expected one compiled attempt");
        assert_eq!(
            metrics.compiled_pipeline_tasks_executed,
            pipeline.tasks().len() as u64,
            "{label}: expected all compiled tasks to execute"
        );
        assert_eq!(metrics.compiled_pipeline_host_arrow_sinks, 1, "{label}: expected one host Arrow sink");
    }

    remove_d3_6_contract_parquets();
}

#[test]
fn execute_query_plan_routes_root_hash_join_through_compiled_pipeline() {
    let Ok(runtime) = cudf::Runtime::new() else {
        return;
    };
    let dir = unique_temp_dir("cudf-query-engine-compiled-d15-root");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    write_join_parquets(&left_path, &right_path);

    let plan = root_join_data_plan(&left_path, &right_path);
    let analysis = plan.capability_analysis();
    let pipeline = compile_pipeline(&plan, &analysis).expect("root join should compile");
    let context = bounded_context(Arc::new(runtime));

    let result = execute_query_plan(&context, &plan).expect("root join should execute through compiled scheduler");
    let batch = result.output().as_host_arrow().expect("compiled output should be host Arrow");
    let metrics = result.metrics();

    assert_eq!(int32_values(batch, 0), vec![Some(2), Some(4)]);
    assert_eq!(int32_values(batch, 1), vec![Some(200), Some(400)]);
    assert_eq!(metrics.compiled_pipeline_attempts, 1);
    assert_eq!(metrics.compiled_pipeline_tasks_executed, pipeline.tasks().len() as u64);
    assert_eq!(metrics.compiled_pipeline_host_arrow_sinks, 1);
    assert_eq!(metrics.hash_join_build_count, 1);
}

#[test]
fn execute_query_plan_routes_selected_adaptive_root_hash_join_through_adaptive_shell() {
    let Ok(runtime) = cudf::Runtime::new() else {
        return;
    };
    let dir = unique_temp_dir("cudf-query-engine-adaptive-e5-root-join");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    write_join_parquets(&left_path, &right_path);
    let plan = root_join_data_plan(&left_path, &right_path);
    let memory_history = Arc::new(AdaptiveMemoryHistory::new());
    memory_history.record_observation(
        AdaptiveMemoryHistoryKey::for_query_plan(&plan),
        AdaptiveMemoryObservation {
            resident_dataframe_peak_bytes: Some(2048),
            reservation_granted_bytes: 4096,
            source_chunks_committed: 2,
        },
    );
    let context = bounded_context_with_mode_and_memory_history(
        Arc::new(runtime),
        NativePipelineExecutionMode::Adaptive,
        Arc::clone(&memory_history),
    );

    assert_eq!(context.native_pipeline_execution_mode(), NativePipelineExecutionMode::Adaptive);
    let result = execute_query_plan(&context, &plan).expect("adaptive selected root join should execute");
    let batch = result.output().as_host_arrow().expect("adaptive output should be host Arrow");
    let metrics = result.metrics();

    assert_eq!(batch.num_rows(), 2);
    assert_eq!(int32_values(batch, 0), vec![Some(2), Some(4)]);
    assert_eq!(int32_values(batch, 1), vec![Some(200), Some(400)]);
    assert_eq!(metrics.compiled_pipeline_attempts, 1);
    assert!(metrics.compiled_pipeline_tasks_executed > 0);
    assert_eq!(metrics.compiled_pipeline_host_arrow_sinks, 1);
    assert_eq!(metrics.adaptive_retry_count, 0);
    assert_eq!(metrics.adaptive_spill_count, 0);
    assert_eq!(metrics.adaptive_downgrade_count, 0);
    assert_eq!(metrics.adaptive_memory_history_lookup_count, 1);
    assert_eq!(metrics.adaptive_memory_history_hit_count, 1);
    assert_eq!(metrics.adaptive_memory_history_miss_count, 0);
}

#[test]
fn execute_query_plan_uses_adaptive_task_history_to_lower_repeated_reservations() {
    let Ok(runtime) = cudf::Runtime::new() else {
        return;
    };
    let runtime = Arc::new(runtime);
    let dir = unique_temp_dir("cudf-query-engine-adaptive-w5-task-history");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    write_join_parquets(&left_path, &right_path);
    let plan = root_join_data_plan(&left_path, &right_path);
    let analysis = plan.capability_analysis();
    let pipeline = compile_pipeline(&plan, &analysis).expect("root join should compile");

    let baseline_context = bounded_context_with_mode(Arc::clone(&runtime), NativePipelineExecutionMode::Adaptive);
    let baseline_result =
        execute_query_plan(&baseline_context, &plan).expect("baseline adaptive run should collect task metrics");
    let baseline_metrics = baseline_result.metrics();
    let (refined_task_id, baseline_requested_bytes) = baseline_metrics
        .task_runtime_metrics
        .iter()
        .find_map(|(task_id, row)| {
            (row.operator_kind == PipelineTaskKind::JoinProbe.as_str() && row.reservation_requested_bytes > 8)
                .then_some((task_id.clone(), row.reservation_requested_bytes))
        })
        .expect("baseline join probe task should request enough bytes to demonstrate downward refinement");
    let history = Arc::new(AdaptiveMemoryHistory::new());
    history.record_observation(
        AdaptiveMemoryHistoryKey::for_query_plan(&plan),
        AdaptiveMemoryObservation {
            resident_dataframe_peak_bytes: baseline_metrics.resident_dataframe_peak_bytes,
            reservation_granted_bytes: baseline_metrics.reservation_granted_bytes,
            source_chunks_committed: baseline_metrics.source_chunks_committed,
        },
    );
    history.record_task_observations_from_query_metrics(&plan, &pipeline, baseline_metrics);

    let refined_context = bounded_context_with_mode_and_memory_history(
        Arc::clone(&runtime),
        NativePipelineExecutionMode::Adaptive,
        Arc::clone(&history),
    );
    let refined_result =
        execute_query_plan(&refined_context, &plan).expect("adaptive task-history run should still execute");
    let refined_metrics = refined_result.metrics();
    let refined_requested_bytes = refined_metrics
        .task_runtime_metrics
        .get(&refined_task_id)
        .expect("refined run should report the same structural compiled task")
        .reservation_requested_bytes;

    assert!(
        refined_requested_bytes < baseline_requested_bytes,
        "adaptive task history should lower repeated task reservation requests: baseline={baseline_requested_bytes}, refined={refined_requested_bytes}"
    );
    assert_eq!(refined_metrics.adaptive_memory_history_lookup_count, 1);
    assert_eq!(refined_metrics.adaptive_memory_history_hit_count, 1);
    assert_eq!(refined_metrics.adaptive_memory_history_miss_count, 0);
}

#[test]
fn execute_query_plan_rejects_selected_adaptive_unsupported_join_with_adaptive_labels() {
    let Ok(runtime) = cudf::Runtime::new() else {
        return;
    };
    let dir = unique_temp_dir("cudf-query-engine-adaptive-e5-unsupported");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    write_join_parquets(&left_path, &right_path);
    let plan = root_join_gpu_sink_data_plan(&left_path, &right_path);
    let context = bounded_context_with_mode(Arc::new(runtime), NativePipelineExecutionMode::Adaptive);

    let error = match execute_query_plan(&context, &plan) {
        Ok(_) => panic!("unsupported adaptive graph should be rejected"),
        Err(error) => error,
    };
    let rendered = error.to_string();

    assert_eq!(error.code(), cudf_query_engine::QueryErrorCode::UnsupportedPlanShape);
    assert_eq!(error.fact_value("execution_mode"), Some("unsupported_adaptive"));
    assert_eq!(error.fact_value("candidate_status"), Some("rejected"));
    assert_eq!(error.fact_value("adaptive_strategy"), Some("unsupported"));
    assert_eq!(error.fact_value("runtime_actions_allowed"), Some("none"));
    assert_eq!(error.fact_value("retry_count"), Some("0"));
    assert_eq!(error.fact_value("spill_count"), Some("0"));
    assert_eq!(error.fact_value("downgrade_count"), Some("0"));
    assert_eq!(error.fact_value("rejection_reason"), Some("unsupported_sink_boundary"));
    assert_eq!(error.fact_value("strict_bounded_contract_report"), Some("not_emitted"));
    assert!(rendered.contains("execution_mode=unsupported_adaptive"));
    assert!(!rendered.contains("execution_mode=strict_bounded"));
    assert!(!rendered.contains("execution_mode=unsupported_strict_bounded"));
}

#[test]
fn execute_query_plan_routes_keyless_root_join_strategy_through_compiled_pipeline() {
    let Ok(runtime) = cudf::Runtime::new() else {
        return;
    };
    let dir = unique_temp_dir("cudf-query-engine-compiled-d34-keyless-root");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    write_join_parquets(&left_path, &right_path);
    let plan = keyless_root_join_data_plan(&left_path, &right_path);
    let analysis = plan.capability_analysis();
    assert!(analysis.admits_root_join());
    let pipeline = compile_pipeline(&plan, &analysis).expect("keyless root join should compile");
    assert_eq!(
        compiled_bounded_query_plan_support(&plan, &pipeline).expect("plan-aware support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );
    let context = bounded_context(Arc::new(runtime));

    let result = execute_query_plan(&context, &plan).expect("D.3.4 keyless root join should execute");
    let batch = result.output().as_host_arrow().expect("compiled output should be host Arrow");
    let metrics = result.metrics();

    assert_eq!(batch.num_rows(), 8);
    assert_eq!(metrics.compiled_pipeline_attempts, 1);
    assert_eq!(metrics.compiled_pipeline_tasks_executed, pipeline.tasks().len() as u64);
    assert_eq!(metrics.compiled_pipeline_host_arrow_sinks, 1);
    assert_eq!(metrics.hash_join_build_count, 0);
    assert!(
        metrics
            .operators
            .iter()
            .filter_map(|operator| operator.join.as_ref())
            .any(|join| join.execution_strategy == "bounded_keyless_conditional_join")
    );
}

#[test]
fn execute_query_plan_routes_root_hash_join_with_unary_inputs_through_compiled_pipeline() {
    let Ok(runtime) = cudf::Runtime::new() else {
        return;
    };
    let dir = unique_temp_dir("cudf-query-engine-compiled-d15-unary-join");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    write_join_parquets(&left_path, &right_path);

    let plan = root_join_with_unary_inputs_data_plan(&left_path, &right_path);
    let analysis = plan.capability_analysis();
    let pipeline = compile_pipeline(&plan, &analysis).expect("root join with unary inputs should compile");
    assert_eq!(
        compiled_bounded_pipeline_support(&pipeline).expect("support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );
    let context = bounded_context(Arc::new(runtime));

    let result = execute_query_plan(&context, &plan).expect("root join with unary inputs should execute");
    let batch = result.output().as_host_arrow().expect("compiled output should be host Arrow");
    let metrics = result.metrics();

    assert_eq!(int32_values(batch, 0), vec![Some(2), Some(4)]);
    assert_eq!(int32_values(batch, 1), vec![Some(200), Some(400)]);
    assert_eq!(metrics.compiled_pipeline_attempts, 1);
    assert_eq!(metrics.compiled_pipeline_tasks_executed, pipeline.tasks().len() as u64);
    assert_eq!(metrics.compiled_pipeline_host_arrow_sinks, 1);
    assert_eq!(metrics.hash_join_build_count, 1);
}

#[test]
fn execute_query_plan_leaves_unsupported_compiled_graphs_on_legacy_fallback() {
    let Ok(runtime) = cudf::Runtime::new() else {
        return;
    };
    let dir = unique_temp_dir("cudf-query-engine-compiled-d15-fallback");
    let path = dir.join("input.parquet");
    let source_schema = write_source_unary_parquet(&path);
    let plan = aggregate_sort_limit_data_plan(&path, source_schema);
    let context = bounded_context(Arc::new(runtime));

    let result =
        execute_query_plan(&context, &plan).expect("unsupported D.1 compiled graph should use legacy fallback");
    let batch = result.output().as_host_arrow().expect("fallback output should be host Arrow");
    let metrics = result.metrics();

    assert_eq!(batch.num_rows(), 4);
    assert_eq!(metrics.compiled_pipeline_attempts, 0);
    assert_eq!(metrics.compiled_pipeline_tasks_executed, 0);
    assert_eq!(metrics.compiled_pipeline_host_arrow_sinks, 0);
}

#[test]
fn execute_query_plan_rejects_admitted_join_when_compiled_runtime_cannot_own_sink() {
    let Ok(runtime) = cudf::Runtime::new() else {
        return;
    };
    let dir = unique_temp_dir("cudf-query-engine-compiled-d37-required");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    write_join_parquets(&left_path, &right_path);
    let plan = root_join_gpu_sink_data_plan(&left_path, &right_path);
    let analysis = plan.capability_analysis();
    assert!(analysis.admits_root_join());
    let context = bounded_context(Arc::new(runtime));

    let error = match execute_query_plan(&context, &plan) {
        Ok(_) => panic!("capability-admitted bounded joins must not fall back to legacy path dispatch"),
        Err(error) => error,
    };
    let rendered = error.to_string();

    assert_eq!(error.code(), cudf_query_engine::QueryErrorCode::UnsupportedPlanShape);
    assert_eq!(error.fact_value("compiled_pipeline_required"), Some("true"));
    assert_eq!(error.fact_value("compiled_pipeline_rejection_reason"), Some("unsupported_sink_boundary"));
    assert_eq!(error.fact_value("legacy_bounded_join_path_bridge"), Some("disabled"));
    assert_eq!(error.fact_value("result_sink_kind"), Some("gpu_resident"));
    assert_no_legacy_execution_selection_terms("D.3.7 compiled-required rejection", &rendered);
}

#[test]
fn execute_query_plan_routes_root_hash_join_aggregate_through_compiled_pipeline() {
    let Ok(runtime) = cudf::Runtime::new() else {
        return;
    };
    let dir = unique_temp_dir("cudf-query-engine-compiled-d23-aggregate");
    let customers_path = dir.join("customers.parquet");
    let orders_path = dir.join("orders.parquet");
    write_single_join_post_root_parquets(&customers_path, &orders_path);
    let plan = single_join_post_root_aggregate_data_plan(&customers_path, &orders_path);
    let analysis = plan.capability_analysis();
    assert!(analysis.admits_single_join_post_root_pipeline());
    let pipeline = compile_pipeline(&plan, &analysis).expect("single-join aggregate plan should compile");
    assert_task_kind_count(&pipeline, PipelineTaskKind::Aggregate, 1);
    assert_task_kind_count(&pipeline, PipelineTaskKind::FinalSort, 0);
    assert_task_kind_count(&pipeline, PipelineTaskKind::Limit, 0);
    assert_eq!(
        compiled_bounded_pipeline_support(&pipeline).expect("support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );
    let context = bounded_context(Arc::new(runtime));

    let result =
        execute_query_plan(&context, &plan).expect("D.2.3 aggregate graph should execute through compiled path");
    let batch = result.output().as_host_arrow().expect("compiled output should be host Arrow");
    let metrics = result.metrics();

    assert_eq!(batch.num_rows(), 3);
    assert_eq!(int32_int64_pairs_sorted(batch, 0, 1), vec![(Some(1), Some(2)), (Some(2), Some(1)), (Some(3), Some(0))]);
    assert_eq!(metrics.compiled_pipeline_attempts, 1);
    assert_eq!(metrics.compiled_pipeline_tasks_executed, pipeline.tasks().len() as u64);
    assert_eq!(metrics.compiled_pipeline_host_arrow_sinks, 1);
    assert_eq!(metrics.hash_join_build_count, 1);
    assert_compiled_partial_aggregate_uses_work_reservation_transfer(metrics);
}

#[test]
fn execute_query_plan_sizes_compiled_stream_aggregate_string_group_keys_inside_reservations() {
    let Ok(runtime) = cudf::Runtime::new() else {
        return;
    };
    let dir = unique_temp_dir("cudf-query-engine-compiled-string-group-aggregate");
    let customers_path = dir.join("customers.parquet");
    let orders_path = dir.join("orders.parquet");
    write_string_group_join_post_root_parquets(&customers_path, &orders_path);
    let plan = string_group_join_post_root_aggregate_data_plan(&customers_path, &orders_path);
    let analysis = plan.capability_analysis();
    let pipeline = compile_pipeline(&plan, &analysis).expect("string grouped aggregate plan should compile");
    assert_task_kind_count(&pipeline, PipelineTaskKind::Aggregate, 1);
    assert_eq!(
        compiled_bounded_query_plan_support(&plan, &pipeline).expect("plan-aware support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );
    let context = bounded_context(Arc::new(runtime));

    let result = execute_query_plan(&context, &plan)
        .expect("compiled stream aggregate should size string group keys inside reservations");
    let batch = result.output().as_host_arrow().expect("compiled output should be host Arrow");
    let metrics = result.metrics();

    assert_eq!(batch.num_rows(), 3);
    assert_eq!(
        string_int64_pairs_sorted(batch, 0, 1),
        vec![(Some("A".to_string()), Some(2)), (Some("BBBB".to_string()), Some(1)), (Some("CCC".to_string()), Some(0)),]
    );
    assert_eq!(metrics.compiled_pipeline_attempts, 1);
    assert_eq!(metrics.compiled_pipeline_tasks_executed, pipeline.tasks().len() as u64);
    assert_compiled_partial_aggregate_uses_work_reservation_transfer(metrics);
    assert!(metrics.reservation_events.iter().any(|event| {
        event.memory_boundary == "compiled_aggregate_string_length_sizing"
            && event.action == MemoryReservationEventAction::Reserve
            && event.reservation_kind == ReservationKind::ChunkOperator
    }));
}

fn assert_compiled_partial_aggregate_uses_work_reservation_transfer(metrics: &QueryMetrics) {
    assert!(metrics.partial_aggregate_count > 0);
    assert!(metrics.reservation_events.iter().any(|event| {
        event.memory_boundary == "compiled_aggregate_partial_state_work"
            && event.action == MemoryReservationEventAction::Reserve
            && event.reservation_kind == ReservationKind::PartialState
    }));
    assert!(metrics.reservation_events.iter().any(|event| {
        event.memory_boundary == "compiled_aggregate_partial_state"
            && event.action == MemoryReservationEventAction::Transfer
            && event.reservation_kind == ReservationKind::PartialState
    }));
    assert!(!metrics.reservation_events.iter().any(|event| {
        event.memory_boundary == "compiled_aggregate_partial_state"
            && event.action == MemoryReservationEventAction::Reserve
            && event.reservation_kind == ReservationKind::PartialState
    }));
}

#[test]
fn execute_query_plan_routes_root_hash_join_final_sort_limit_through_compiled_pipeline() {
    let Ok(runtime) = cudf::Runtime::new() else {
        return;
    };
    let dir = unique_temp_dir("cudf-query-engine-compiled-d24-sort-limit");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    write_join_parquets(&left_path, &right_path);
    let plan = root_join_sort_limit_data_plan(&left_path, &right_path);
    let analysis = plan.capability_analysis();
    let pipeline = compile_pipeline(&plan, &analysis).expect("root join final-sort/limit plan should compile");
    assert_task_kind_count(&pipeline, PipelineTaskKind::Aggregate, 0);
    assert_task_kind_count(&pipeline, PipelineTaskKind::FinalSort, 1);
    assert_task_kind_count(&pipeline, PipelineTaskKind::Limit, 1);
    assert_eq!(
        compiled_bounded_pipeline_support(&pipeline).expect("support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );
    let final_sort_gate =
        runtime_visible_gate_for_task_kind(&pipeline, PipelineTaskKind::FinalSort, ReservationKind::FinalSort)
            .expect("compiled final-sort task should retain its modeled reservation gate");
    assert_eq!(final_sort_gate.status(), ContractStatus::Modeled);
    assert_eq!(final_sort_gate.quality(), ContractQuality::HardBudgetedReportWithUnresolvedPreflight);
    let context = bounded_context(Arc::new(runtime));

    let result = execute_query_plan(&context, &plan).expect("D.2.4 final-sort/limit graph should execute");
    let batch = result.output().as_host_arrow().expect("compiled output should be host Arrow");
    let metrics = result.metrics();

    assert_eq!(batch.num_rows(), 1);
    assert_eq!(int32_values(batch, 0), vec![Some(4)]);
    assert_eq!(int32_values(batch, 1), vec![Some(400)]);
    assert_eq!(metrics.compiled_pipeline_attempts, 1);
    assert_eq!(metrics.compiled_pipeline_tasks_executed, pipeline.tasks().len() as u64);
    assert_eq!(metrics.compiled_pipeline_host_arrow_sinks, 1);
    assert_eq!(metrics.hash_join_build_count, 1);
}

#[test]
fn execute_query_plan_routes_single_join_post_root_d2_graph_through_compiled_pipeline() {
    let Ok(runtime) = cudf::Runtime::new() else {
        return;
    };
    let dir = unique_temp_dir("cudf-query-engine-compiled-d22-single-join");
    let customers_path = dir.join("customers.parquet");
    let orders_path = dir.join("orders.parquet");
    write_single_join_post_root_parquets(&customers_path, &orders_path);
    let plan = single_join_post_root_aggregate_sort_limit_data_plan(&customers_path, &orders_path);
    let analysis = plan.capability_analysis();
    assert!(analysis.admits_single_join_post_root_pipeline());
    let pipeline = compile_pipeline(&plan, &analysis).expect("single-join post-root plan should compile");
    assert_eq!(
        compiled_bounded_pipeline_support(&pipeline).expect("support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );
    let context = bounded_context(Arc::new(runtime));

    let result = execute_query_plan(&context, &plan).expect("D.2 graph should execute through compiled path");
    let batch = result.output().as_host_arrow().expect("compiled output should be host Arrow");
    let metrics = result.metrics();

    assert_eq!(batch.num_rows(), 2);
    assert_eq!(int32_values(batch, 0), vec![Some(3), Some(2)]);
    assert_eq!(int64_values(batch, 1), vec![Some(0), Some(1)]);
    assert_eq!(metrics.compiled_pipeline_attempts, 1);
    assert_eq!(metrics.compiled_pipeline_tasks_executed, pipeline.tasks().len() as u64);
    assert_eq!(metrics.compiled_pipeline_host_arrow_sinks, 1);
    assert_eq!(metrics.hash_join_build_count, 1);
}

#[test]
fn execute_query_plan_routes_simple_multi_join_tree_through_compiled_pipeline() {
    let Ok(runtime) = cudf::Runtime::new() else {
        return;
    };
    let dir = unique_temp_dir("cudf-query-engine-compiled-d26-multi-join");
    let a_path = dir.join("a.parquet");
    let b_path = dir.join("b.parquet");
    let c_path = dir.join("c.parquet");
    write_multi_join_parquets(&a_path, &b_path, &c_path);
    let plan = simple_multi_join_plan(&a_path, &b_path, &c_path);
    let analysis = plan.capability_analysis();
    assert!(analysis.admits_closed_multi_join_subgraph());
    let pipeline = compile_pipeline(&plan, &analysis).expect("simple multi-join plan should compile");
    assert_eq!(
        compiled_bounded_pipeline_support(&pipeline).expect("support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );
    let context = bounded_context(Arc::new(runtime));

    let result = execute_query_plan(&context, &plan).expect("D.2.6 multi-join graph should execute");
    let batch = result.output().as_host_arrow().expect("compiled output should be host Arrow");
    let metrics = result.metrics();

    assert_eq!(batch.num_rows(), 2);
    assert_eq!(
        int32_quads_sorted(batch, 0, 1, 2, 3),
        vec![(Some(2), Some(20), Some(200), Some(2000)), (Some(3), Some(30), Some(300), Some(3000)),]
    );
    assert_eq!(metrics.compiled_pipeline_attempts, 1);
    assert_eq!(metrics.compiled_pipeline_tasks_executed, pipeline.tasks().len() as u64);
    assert_eq!(metrics.compiled_pipeline_host_arrow_sinks, 1);
    assert_eq!(metrics.hash_join_build_count, 2);
}

#[test]
fn compiled_host_sink_diagnostics_are_structured_without_legacy_routing_terms() {
    let Ok(runtime) = cudf::Runtime::new() else {
        return;
    };
    let plan = root_join_plan();
    let analysis = plan.capability_analysis();
    let pipeline = compile_pipeline(&plan, &analysis).expect("root join should compile");
    let context = bounded_context(Arc::new(runtime));
    let gpu_inputs = cudf_query_engine::GpuResidentInputMap::new();
    let mut metrics = QueryMetrics::default();
    let mut executor = CompiledGpuTaskExecutor::new(&context, &plan, &pipeline, &gpu_inputs, &mut metrics).unwrap();

    let error = executor.take_host_output().expect_err("empty host sink should be a structured invariant error");
    let rendered = error.to_string();

    assert_eq!(error.code(), cudf_query_engine::QueryErrorCode::InvalidPlan);
    assert_eq!(error.facts().get("compiled_task_id"), Some("t5"));
    assert_eq!(error.facts().get("sink_boundary_kind"), Some("host_arrow_materialization"));
    assert!(!rendered.contains("BoundedJoinExecutionFamily"));
    assert!(!rendered.contains("BoundedJoinExecutionPath"));
    assert!(!rendered.contains("bounded_join_execution_path"));
}
