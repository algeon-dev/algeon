use super::*;

#[test]
fn compiled_gpu_executor_runs_root_hash_join_build_probe_materialize() {
    let Ok(runtime) = cudf::Runtime::new() else {
        return;
    };
    let dir = unique_temp_dir("nexus-query-engine-compiled-d14");
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
    let gpu_inputs = nexus_query_engine::runtime::resident::GpuResidentInputMap::new();

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
    let dir = unique_temp_dir("nexus-query-engine-compiled-d32-shared-source");
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
    let gpu_inputs = nexus_query_engine::runtime::resident::GpuResidentInputMap::new();

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
    assert_eq!(metrics.source_chunks_committed, 2);
    assert_eq!(compiled_batch.num_rows(), 3);
    assert_eq!(int32_values(&compiled_batch, 0), vec![Some(1), Some(2), Some(4)]);
    assert_eq!(int32_values(&compiled_batch, 1), vec![Some(10), Some(20), Some(40)]);
}

#[test]
fn execute_query_plan_rejects_grouped_aggregate_join_build_input_without_strict_envelope() {
    let Ok(runtime) = cudf::Runtime::new() else {
        return;
    };
    let dir = unique_temp_dir("nexus-query-engine-compiled-d33-aggregate-join-input");
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

    let error = match execute_query_plan(&context, &plan) {
        Ok(_) => panic!("D.3.3 grouped aggregate join-build input needs explicit strict envelope admission"),
        Err(error) => error,
    };
    assert_eq!(error.code(), nexus_query_engine::error::QueryErrorCode::UnsupportedPlanShape);
    assert_eq!(error.fact_value("bounded_join_rejection_boundary"), Some("admission_capability"));
    assert_eq!(
        error.fact_value("bounded_join_rejection_reason"),
        Some("source_leaf_pipeline_requires_source_filter_projection")
    );
    assert_eq!(error.fact_value("bounded_join_rejection_category"), Some("missing_subgraph_invariant"));
}

#[test]
fn compiled_gpu_executor_streams_grouped_aggregate_join_build_input_source() {
    let Ok(runtime) = cudf::Runtime::new() else {
        return;
    };
    let dir = unique_temp_dir("nexus-query-engine-compiled-aggregate-join-input-stream");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    write_aggregate_join_input_parquets(&left_path, &right_path);

    let optimizer_options = NativeOptimizerOptions {
        join_build_side_selection: JoinBuildSideSelectionOptions::disabled(),
        ..NativeOptimizerOptions::default()
    };
    let plan =
        optimize_query_plan(aggregate_join_build_input_data_plan(&left_path, &right_path), &optimizer_options).plan;
    let analysis = plan.capability_analysis();
    let pipeline = compile_pipeline(&plan, &analysis).expect("aggregate join input plan should compile");
    assert_eq!(
        compiled_bounded_query_plan_support(&plan, &pipeline).expect("plan-aware support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );
    let scheduler = CompiledPipelineScheduler::new(&pipeline).expect("pipeline should be schedulable");
    let context = bounded_context(Arc::new(runtime));
    let mut metrics = QueryMetrics::default();
    let gpu_inputs = GpuResidentInputMap::new();

    let (state, batch) = {
        let mut executor = CompiledGpuTaskExecutor::new(&context, &plan, &pipeline, &gpu_inputs, &mut metrics).unwrap();
        let state = scheduler.run(&mut executor).expect("compiled aggregate join input should execute");
        let output = executor.take_host_output().expect("compiled host sink should materialize");
        (state, output)
    };

    assert_eq!(state.live_resource_count(), 0);
    assert_eq!(int32_int64_pairs_sorted(&batch, 0, 1), vec![(Some(1), Some(15)), (Some(2), Some(20))]);
    assert_eq!(
        metrics.partial_aggregate_count, 2,
        "the aggregate input source has two row-group chunks and should be reduced as chunk-local partials"
    );
    assert_compiled_partial_aggregate_uses_work_reservation_transfer(&metrics);
    assert_compiled_final_merge_uses_partitioned_strategy(&metrics);
}

#[test]
fn compiled_gpu_executor_streams_shared_source_for_join_build_and_grouped_aggregate() {
    let Ok(runtime) = cudf::Runtime::new() else {
        return;
    };
    let dir = unique_temp_dir("nexus-query-engine-compiled-shared-source-aggregate");
    let fact_path = dir.join("fact.parquet");
    let dim_path = dir.join("dim.parquet");
    write_aggregate_join_input_parquets(&fact_path, &dim_path);

    let optimizer_options = NativeOptimizerOptions {
        join_build_side_selection: JoinBuildSideSelectionOptions::disabled(),
        ..NativeOptimizerOptions::default()
    };
    let plan = optimize_query_plan(
        shared_source_join_build_and_aggregate_data_plan(&fact_path, &dim_path),
        &optimizer_options,
    )
    .plan;
    let analysis = plan.capability_analysis();
    let pipeline = compile_pipeline(&plan, &analysis).expect("shared-source aggregate plan should compile");
    assert!(
        pipeline.debug_render().contains("kind=shared_resident_producer"),
        "shared source must remain represented as one fan-out resource"
    );
    assert_eq!(
        compiled_bounded_query_plan_support(&plan, &pipeline).expect("plan-aware support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );
    let scheduler = CompiledPipelineScheduler::new(&pipeline).expect("pipeline should be schedulable");
    let context = bounded_context(Arc::new(runtime));
    let mut metrics = QueryMetrics::default();
    let gpu_inputs = GpuResidentInputMap::new();

    let (state, batch) = {
        let mut executor = CompiledGpuTaskExecutor::new(&context, &plan, &pipeline, &gpu_inputs, &mut metrics).unwrap();
        let state = scheduler.run(&mut executor).expect("shared source aggregate should execute");
        let output = executor.take_host_output().expect("compiled host sink should materialize");
        (state, output)
    };

    assert_eq!(state.live_resource_count(), 0);
    assert_eq!(
        int32_int64_pairs_sorted(&batch, 0, 1),
        vec![(Some(1), Some(15)), (Some(1), Some(15)), (Some(2), Some(20))]
    );
    assert_eq!(
        metrics.partial_aggregate_count, 2,
        "the shared aggregate branch should clone the deferred source stream and reduce both row-group chunks"
    );
    assert_compiled_final_merge_uses_partitioned_strategy(&metrics);
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
    let dir = unique_temp_dir("nexus-query-engine-compiled-d15-root");
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
    let dir = unique_temp_dir("nexus-query-engine-adaptive-e5-root-join");
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
    let dir = unique_temp_dir("nexus-query-engine-adaptive-w5-task-history");
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
fn execute_query_plan_routes_gpu_sink_join_through_adaptive_compiled_pipeline() {
    let Ok(runtime) = cudf::Runtime::new() else {
        return;
    };
    let dir = unique_temp_dir("nexus-query-engine-adaptive-e5-unsupported");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    write_join_parquets(&left_path, &right_path);
    let plan = root_join_gpu_sink_data_plan(&left_path, &right_path);
    let context = bounded_context_with_mode(Arc::new(runtime), NativePipelineExecutionMode::Adaptive);

    let result = execute_query_plan(&context, &plan).expect("adaptive compiled GPU sink join should execute");
    let QueryOutput::GpuResident(handle) = result.output() else {
        panic!("GPU sink should keep the compiled output resident");
    };
    let mut materialization_metrics = QueryMetrics::default();
    let batch = materialize_gpu_result_to_host(handle.clone(), &mut materialization_metrics)
        .expect("resident compiled output should materialize for assertions");

    assert_eq!(batch.num_rows(), 2);
    assert_eq!(int32_values(&batch, 0), vec![Some(2), Some(4)]);
    assert_eq!(int32_values(&batch, 1), vec![Some(200), Some(400)]);
    assert_eq!(result.metrics().compiled_pipeline_attempts, 1);
    assert_eq!(result.metrics().compiled_pipeline_host_arrow_sinks, 0);
}

#[test]
fn execute_query_plan_routes_keyless_root_join_strategy_through_compiled_pipeline() {
    let Ok(runtime) = cudf::Runtime::new() else {
        return;
    };
    let dir = unique_temp_dir("nexus-query-engine-compiled-d34-keyless-root");
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
    let dir = unique_temp_dir("nexus-query-engine-compiled-d15-unary-join");
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
    let dir = unique_temp_dir("nexus-query-engine-compiled-d15-fallback");
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
fn execute_query_plan_routes_gpu_resident_sink_join_through_compiled_pipeline() {
    let Ok(runtime) = cudf::Runtime::new() else {
        return;
    };
    let dir = unique_temp_dir("nexus-query-engine-compiled-d37-required");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    write_join_parquets(&left_path, &right_path);
    let plan = root_join_gpu_sink_data_plan(&left_path, &right_path);
    let analysis = plan.capability_analysis();
    assert!(analysis.admits_root_join());
    let admission = strict_bounded_pipeline_admission(&plan)
        .expect("strict admission should not fail")
        .expect("GPU sink root join should be strict-admitted");
    assert_eq!(admission.pipeline().sink().kind(), SinkBoundaryKind::GpuResidentOutput);
    assert!(
        admission.pipeline().gates().iter().all(|gate| {
            gate.reservation_kind() != ReservationKind::Materialization
                && gate.memory_boundary() != "host_materialization"
                && gate.required_contract() != "query_engine_explicit_sink_materialization_reservation"
        }),
        "GPU sink compiled pipeline must not carry host materialization gates"
    );
    assert!(
        nexus_query_engine::admission::strict_bounded_pipeline_rejection_projection(&plan)
            .expect("strict rejection projection should not fail")
            .is_none(),
        "GPU sink should stay inside the compiled bounded envelope"
    );
    assert!(
        nexus_query_engine::admission::bounded_join_modeled_facts(
            &plan,
            nexus_query_engine::runtime::ExecutionBoundaryPolicy::default()
        )
        .is_some(),
        "GPU sink plans admitted by strict compiled execution must emit modeled bounded-join facts"
    );
    assert!(
        nexus_query_engine::admission::bounded_join_unsupported_shape_facts(
            &plan,
            nexus_query_engine::runtime::ExecutionBoundaryPolicy::default()
        )
        .is_none(),
        "GPU sink plans admitted by strict compiled execution must not emit unsupported bounded-join facts"
    );
    let context = bounded_context(Arc::new(runtime));

    let result = execute_query_plan(&context, &plan).expect("strict compiled GPU sink join should execute");
    let QueryOutput::GpuResident(handle) = result.output() else {
        panic!("GPU sink should keep the compiled output resident");
    };
    let mut materialization_metrics = QueryMetrics::default();
    let batch = materialize_gpu_result_to_host(handle.clone(), &mut materialization_metrics)
        .expect("resident compiled output should materialize for assertions");

    assert_eq!(batch.num_rows(), 2);
    assert_eq!(int32_values(&batch, 0), vec![Some(2), Some(4)]);
    assert_eq!(int32_values(&batch, 1), vec![Some(200), Some(400)]);
    assert_eq!(result.metrics().compiled_pipeline_attempts, 1);
    assert_eq!(result.metrics().compiled_pipeline_host_arrow_sinks, 0);

    let mut passthrough_inputs = GpuResidentInputMap::new();
    passthrough_inputs.insert("compiled_root_join".to_string(), handle.clone());
    let mut passthrough_builder = QueryPlanBuilder::new();
    let passthrough_source = passthrough_builder.add_source(SourceNode::GpuResident(GpuResidentSource::new(
        "compiled_root_join",
        plan.output_schema().clone(),
    )));
    let passthrough_plan = passthrough_builder.finish_with_host_sink(passthrough_source);
    let passthrough = execute_query_plan_with_gpu_inputs(&context, &passthrough_plan, &passthrough_inputs)
        .expect("compiled GPU output handle should feed a later GPU-resident source");
    let QueryOutput::HostArrow(passthrough_batch) = passthrough.output() else {
        panic!("passthrough consumer should materialize the resident source for assertions");
    };
    assert_eq!(passthrough_batch.num_rows(), 2);
    assert_eq!(int32_values(passthrough_batch, 0), vec![Some(2), Some(4)]);
    assert_eq!(int32_values(passthrough_batch, 1), vec![Some(200), Some(400)]);
}

#[test]
fn pipeline_validation_rejects_non_sink_edge_gate_on_sink_boundary() {
    let plan = root_join_gpu_sink_data_plan(
        Path::new("/tmp/compiled-gpu-sink-invalid-gate-left.parquet"),
        Path::new("/tmp/compiled-gpu-sink-invalid-gate-right.parquet"),
    );
    let analysis = plan.capability_analysis();
    let pipeline = compile_pipeline(&plan, &analysis).expect("GPU sink root join should compile");
    let invalid_gate = ReservationGateId::new(pipeline.gates().len());
    let mut gates = pipeline.gates().to_vec();
    gates.push(ReservationGate::new(
        invalid_gate,
        ReservationGateOwner::SinkBoundary,
        ReservationKind::ProbeSide,
        ReservationRequirementScope::Edge,
        "join_probe_side",
        "query_engine_probe_side_source_reservation",
        ContractStatus::Modeled,
        ContractQuality::QueryEngineOwnedOrdering,
    ));
    let mut sink_gates = pipeline.sink().gates().to_vec();
    sink_gates.push(invalid_gate);
    let malformed = CompiledPipeline::new(
        pipeline.tasks().to_vec(),
        pipeline.edges().to_vec(),
        pipeline.barriers().to_vec(),
        pipeline.resources().to_vec(),
        gates,
        SinkBoundary::new(pipeline.sink().kind(), pipeline.sink().input()).with_gates(sink_gates),
    );

    let error = malformed.validate().expect_err("unrelated edge gates must not attach to the sink boundary");

    assert_eq!(error.code(), nexus_query_engine::error::QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("gate_owner"), Some("sink_boundary"));
    assert_eq!(error.fact_value("requirement_scope"), Some("edge"));
    assert_eq!(error.fact_value("reservation_kind"), Some("probe_side"));
}

#[test]
fn compiled_support_rejects_plan_pipeline_sink_mismatch() {
    let plan = root_join_data_plan(
        Path::new("/tmp/compiled-sink-mismatch-left.parquet"),
        Path::new("/tmp/compiled-sink-mismatch-right.parquet"),
    );
    let gpu_plan = plan.clone_with_gpu_resident_sink();
    let host_pipeline =
        compile_pipeline(&plan, &plan.capability_analysis()).expect("host-sink root join should compile");
    let gpu_pipeline =
        compile_pipeline(&gpu_plan, &gpu_plan.capability_analysis()).expect("GPU-sink root join should compile");

    assert_eq!(
        compiled_bounded_query_plan_support(&gpu_plan, &host_pipeline).expect("support check should not fail"),
        CompiledBoundedPipelineSupport::Unsupported,
        "a host-sink compiled pipeline must not be reused for a GPU-sink query plan"
    );
    assert_eq!(
        compiled_bounded_query_plan_support(&plan, &gpu_pipeline).expect("support check should not fail"),
        CompiledBoundedPipelineSupport::Unsupported,
        "a GPU-sink compiled pipeline must not be reused for a host-sink query plan"
    );
}

#[test]
fn pipeline_validation_rejects_host_sink_output_lifetime() {
    let source = PipelineTaskId::new(0);
    let output = ResidentResourceId::new(0);
    let pipeline =
        single_sink_output_pipeline(SinkBoundaryKind::HostArrowMaterialization, source, vec![(source, output)]);

    let error = pipeline.validate().expect_err("host sinks cannot retain sink-output resources");

    assert_eq!(error.code(), nexus_query_engine::error::QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("resident_resource_id"), Some("r0"));
    assert_eq!(error.fact_value("sink_boundary_kind"), Some("host_arrow_materialization"));
    assert_eq!(error.fact_value("sink_output_resource_count"), Some("1"));
}

#[test]
fn pipeline_validation_rejects_gpu_sink_output_from_wrong_producer() {
    let sink = PipelineTaskId::new(0);
    let producer = PipelineTaskId::new(1);
    let output = ResidentResourceId::new(0);
    let pipeline = single_sink_output_pipeline(SinkBoundaryKind::GpuResidentOutput, sink, vec![(producer, output)]);

    let error = pipeline.validate().expect_err("GPU sink output must be produced by the sink boundary input task");

    assert_eq!(error.code(), nexus_query_engine::error::QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("resident_resource_id"), Some("r0"));
    assert_eq!(error.fact_value("producer_task"), Some("t1"));
    assert_eq!(error.fact_value("sink_boundary_task_id"), Some("t0"));
}

#[test]
fn pipeline_validation_rejects_multiple_gpu_sink_outputs() {
    let sink = PipelineTaskId::new(0);
    let other = PipelineTaskId::new(1);
    let pipeline = single_sink_output_pipeline(
        SinkBoundaryKind::GpuResidentOutput,
        sink,
        vec![(sink, ResidentResourceId::new(0)), (other, ResidentResourceId::new(1))],
    );

    let error = pipeline.validate().expect_err("GPU sinks must expose exactly one sink-output resource");

    assert_eq!(error.code(), nexus_query_engine::error::QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("sink_boundary_task_id"), Some("t0"));
    assert_eq!(error.fact_value("sink_boundary_kind"), Some("gpu_resident_output"));
    assert_eq!(error.fact_value("sink_output_resource_count"), Some("2"));
}

fn single_sink_output_pipeline(
    kind: SinkBoundaryKind,
    sink_input: PipelineTaskId,
    sink_outputs: Vec<(PipelineTaskId, ResidentResourceId)>,
) -> CompiledPipeline {
    let mut task_ids = sink_outputs.iter().map(|(task, _)| *task).collect::<BTreeSet<_>>();
    task_ids.insert(sink_input);
    let tasks = task_ids
        .iter()
        .map(|task| {
            let outputs = sink_outputs
                .iter()
                .filter_map(|(producer, resource)| (*producer == *task).then_some(*resource))
                .collect();
            PipelineTask::new(*task, PipelineTaskKind::SourceRead).with_output_resources(outputs)
        })
        .collect::<Vec<_>>();
    let resources = sink_outputs
        .iter()
        .map(|(producer, resource)| {
            ResidentResource::new(*resource, *producer, vec![], ResidentLifetime::SinkOutput { producer: *producer })
        })
        .collect::<Vec<_>>();
    CompiledPipeline::new(
        tasks,
        vec![],
        vec![
            PipelineBarrier::new(PipelineBarrierId::new(0), PipelineBarrierKind::SinkBoundary)
                .with_upstream_tasks(task_ids.into_iter().collect()),
        ],
        resources,
        vec![],
        SinkBoundary::new(kind, sink_input),
    )
}

#[test]
fn execute_query_plan_routes_root_hash_join_aggregate_through_compiled_pipeline() {
    let Ok(runtime) = cudf::Runtime::new() else {
        return;
    };
    let dir = unique_temp_dir("nexus-query-engine-compiled-d23-aggregate");
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
    let dir = unique_temp_dir("nexus-query-engine-compiled-string-group-aggregate");
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

fn assert_compiled_final_merge_uses_partitioned_strategy(metrics: &QueryMetrics) {
    let decision = metrics
        .aggregate_strategy_decisions
        .iter()
        .find(|decision| {
            decision.facts.get("execution_stage").map(String::as_str) == Some("final_merge")
                && decision.facts.get("selected_strategy").map(String::as_str) == Some("partitioned_merge")
        })
        .expect("compiled final merge should record partitioned final-merge strategy");
    assert_ne!(
        decision.facts.get("partitioned_merge_partition_count").map(String::as_str),
        Some("0"),
        "partitioned final merge should record a positive partition count"
    );
    assert!(metrics.reservation_events.iter().any(|event| {
        event.memory_boundary == "compiled_partitioned_final_aggregate_merge"
            && event.action == MemoryReservationEventAction::Reserve
            && event.reservation_kind == ReservationKind::FinalMerge
    }));
}

#[test]
fn execute_query_plan_routes_root_hash_join_final_sort_limit_through_compiled_pipeline() {
    let Ok(runtime) = cudf::Runtime::new() else {
        return;
    };
    let dir = unique_temp_dir("nexus-query-engine-compiled-d24-sort-limit");
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
    let dir = unique_temp_dir("nexus-query-engine-compiled-d22-single-join");
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
    let dir = unique_temp_dir("nexus-query-engine-compiled-d26-multi-join");
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
fn execute_query_plan_streams_residual_one_shot_probe_from_prior_join() {
    let Ok(runtime) = cudf::Runtime::new() else {
        return;
    };
    let dir = unique_temp_dir("nexus-query-engine-compiled-residual-stream");
    let a_path = dir.join("a.parquet");
    let b_path = dir.join("b.parquet");
    let c_path = dir.join("c.parquet");
    write_residual_stream_join_parquets(&a_path, &b_path, &c_path);
    let plan = residual_one_shot_frame_stream_multi_join_plan(&a_path, &b_path, &c_path);
    let analysis = plan.capability_analysis();
    assert!(analysis.admits_closed_multi_join_subgraph());
    let pipeline = compile_pipeline(&plan, &analysis).expect("residual stream multi-join plan should compile");
    assert_eq!(
        compiled_bounded_pipeline_support(&pipeline).expect("support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );
    let upstream_join = plan
        .nodes()
        .iter()
        .find_map(|node| {
            matches!(&node.kind, PlanNode::Join(join) if join.spec.residual_predicate.is_none()).then_some(node.id)
        })
        .expect("fixture should contain the upstream equi join");
    let residual_join = plan
        .nodes()
        .iter()
        .find_map(|node| {
            matches!(&node.kind, PlanNode::Join(join) if join.spec.residual_predicate.is_some()).then_some(node.id)
        })
        .expect("fixture should contain the downstream residual join");
    let context = bounded_context(Arc::new(runtime));

    let result = execute_query_plan(&context, &plan)
        .expect("residual one-shot stream probe should execute through compiled path");
    let batch = result.output().as_host_arrow().expect("compiled output should be host Arrow");
    let metrics = result.metrics();
    let upstream_join = upstream_join.to_string();
    let residual_join = residual_join.to_string();

    assert_eq!(batch.num_rows(), 1);
    assert_eq!(int32_quads_sorted(batch, 0, 1, 2, 3), vec![(Some(4), Some(40), Some(400), Some(45))]);
    assert_eq!(metrics.compiled_pipeline_attempts, 1);
    assert_eq!(metrics.compiled_pipeline_tasks_executed, pipeline.tasks().len() as u64);
    let join_rows = metrics.operators.iter().filter_map(|operator| operator.join.as_ref()).collect::<Vec<_>>();
    let residual_join_rows = metrics
        .operators
        .iter()
        .filter(|operator| {
            operator.node_id == residual_join
                && operator.join.as_ref().is_some_and(|join| {
                    join.residual_predicate && join.execution_strategy == "bounded_one_shot_equi_join"
                })
        })
        .collect::<Vec<_>>();
    assert_eq!(
        residual_join_rows.len(),
        1,
        "join rows: {join_rows:?}; strategy decisions: {:?}",
        metrics.join_strategy_decisions
    );
    let residual_join_metrics = residual_join_rows[0].join.as_ref().expect("residual join row should be present");
    assert_eq!(residual_join_metrics.probe_rows, 2);
    assert_eq!(residual_join_metrics.output_rows, 1);
    let one_shot_operator_rows =
        join_rows.iter().filter(|join| join.execution_strategy == "bounded_one_shot_equi_join").count();
    assert_eq!(metrics.one_shot_join_count, one_shot_operator_rows as u64);
    let upstream_events = metrics
        .reservation_events
        .iter()
        .filter(|event| event.node_id.as_deref() == Some(upstream_join.as_str()))
        .collect::<Vec<_>>();
    assert!(
        upstream_events
            .iter()
            .filter(|event| {
                (event.memory_boundary == "join_output" || event.memory_boundary == "join_output_stream_chunk")
                    && event.action == MemoryReservationEventAction::Reserve
            })
            .count()
            >= 2,
        "fixture must produce multiple upstream join output chunks: source_chunks_committed={}, source_rows={:?}, upstream_events={upstream_events:?}",
        metrics.source_chunks_committed,
        metrics.source_read_runtime_rows
    );
    assert!(
        !metrics.reservation_events.iter().any(|event| {
            event.node_id.as_deref() == Some(upstream_join.as_str())
                && event.memory_boundary == "join_output_stream_concat"
        }),
        "upstream join output must stay streamed into the residual probe"
    );
    assert!(metrics.reservation_events.iter().any(|event| {
        event.node_id.as_deref() == Some(residual_join.as_str())
            && event.memory_boundary == "join_residual_filter_output"
    }));
    let build_alias_acquires = metrics
        .reservation_events
        .iter()
        .filter(|event| {
            event.node_id.as_deref() == Some(residual_join.as_str())
                && event.memory_boundary == "join_build_side_stream_alias"
                && event.action == MemoryReservationEventAction::AliasAcquire
        })
        .count();
    let build_alias_releases = metrics
        .reservation_events
        .iter()
        .filter(|event| {
            event.node_id.as_deref() == Some(residual_join.as_str())
                && event.memory_boundary == "join_build_side_stream_alias"
                && event.action == MemoryReservationEventAction::AliasRelease
        })
        .count();
    assert!(build_alias_acquires > 0, "residual stream should alias the retained build side");
    assert_eq!(build_alias_acquires, build_alias_releases);
}

#[test]
fn execute_query_plan_streams_residual_one_shot_probe_from_leaf_local_source() {
    let Ok(runtime) = cudf::Runtime::new() else {
        return;
    };
    let dir = unique_temp_dir("nexus-query-engine-compiled-residual-leaf-stream");
    let a_path = dir.join("a.parquet");
    let b_path = dir.join("b.parquet");
    let c_path = dir.join("c.parquet");
    write_residual_stream_join_parquets(&a_path, &b_path, &c_path);
    let plan = residual_one_shot_leaf_stream_join_plan(&a_path, &c_path);
    let analysis = plan.capability_analysis();
    let pipeline = compile_pipeline(&plan, &analysis).expect("residual leaf-stream join plan should compile");
    assert_eq!(
        compiled_bounded_pipeline_support(&pipeline).expect("support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );
    let residual_join = plan
        .nodes()
        .iter()
        .find_map(|node| {
            matches!(&node.kind, PlanNode::Join(join) if join.spec.residual_predicate.is_some()).then_some(node.id)
        })
        .expect("fixture should contain the residual join");
    let context = bounded_context(Arc::new(runtime));

    let result = execute_query_plan(&context, &plan)
        .expect("residual one-shot leaf stream probe should execute through compiled path");
    let batch = result.output().as_host_arrow().expect("compiled output should be host Arrow");
    let metrics = result.metrics();
    let residual_join = residual_join.to_string();

    assert_eq!(batch.num_rows(), 1);
    assert_eq!(int32_triples_sorted(batch, 0, 1, 2), vec![(Some(4), Some(40), Some(45))]);
    assert_eq!(metrics.compiled_pipeline_attempts, 1);
    assert_eq!(metrics.compiled_pipeline_tasks_executed, pipeline.tasks().len() as u64);
    assert!(
        metrics.source_chunks_committed >= 2,
        "fixture must stream the leaf probe source in multiple chunks: source_chunks_committed={}, source_rows={:?}",
        metrics.source_chunks_committed,
        metrics.source_read_runtime_rows
    );
    let join_rows = metrics.operators.iter().filter_map(|operator| operator.join.as_ref()).collect::<Vec<_>>();
    let residual_join_rows = metrics
        .operators
        .iter()
        .filter(|operator| {
            operator.node_id == residual_join
                && operator.join.as_ref().is_some_and(|join| {
                    join.residual_predicate && join.execution_strategy == "bounded_mixed_keyed_residual_join"
                })
        })
        .collect::<Vec<_>>();
    assert!(
        !residual_join_rows.is_empty(),
        "join rows: {join_rows:?}; strategy decisions: {:?}",
        metrics.join_strategy_decisions
    );
    assert_eq!(
        residual_join_rows
            .iter()
            .filter_map(|operator| operator.join.as_ref())
            .map(|join| join.probe_rows)
            .sum::<u64>(),
        4
    );
    assert_eq!(
        residual_join_rows
            .iter()
            .filter_map(|operator| operator.join.as_ref())
            .map(|join| join.output_rows)
            .sum::<u64>(),
        1
    );
    assert_eq!(metrics.one_shot_join_count, 0);
    let build_alias_acquires = metrics
        .reservation_events
        .iter()
        .filter(|event| {
            event.node_id.as_deref() == Some(residual_join.as_str())
                && event.memory_boundary == "join_build_side_stream_alias"
                && event.action == MemoryReservationEventAction::AliasAcquire
        })
        .count();
    let build_alias_releases = metrics
        .reservation_events
        .iter()
        .filter(|event| {
            event.node_id.as_deref() == Some(residual_join.as_str())
                && event.memory_boundary == "join_build_side_stream_alias"
                && event.action == MemoryReservationEventAction::AliasRelease
        })
        .count();
    assert!(build_alias_acquires > 0, "residual stream should alias the retained build side");
    assert_eq!(build_alias_acquires, build_alias_releases);
}

#[test]
fn execute_query_plan_streams_residual_one_shot_right_probe_with_left_build() {
    let Ok(runtime) = cudf::Runtime::new() else {
        return;
    };
    let dir = unique_temp_dir("nexus-query-engine-compiled-residual-left-build-stream");
    let a_path = dir.join("a.parquet");
    let b_path = dir.join("b.parquet");
    let c_path = dir.join("c.parquet");
    write_residual_stream_join_parquets(&a_path, &b_path, &c_path);
    let plan = residual_one_shot_leaf_stream_left_build_join_plan(&a_path, &c_path);
    let analysis = plan.capability_analysis();
    let pipeline =
        compile_pipeline(&plan, &analysis).expect("left-build residual leaf-stream join plan should compile");
    assert_eq!(
        compiled_bounded_pipeline_support(&pipeline).expect("support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );
    let residual_join = plan
        .nodes()
        .iter()
        .find_map(|node| {
            matches!(&node.kind, PlanNode::Join(join) if join.spec.residual_predicate.is_some()).then_some(node.id)
        })
        .expect("fixture should contain the residual join");
    let context = bounded_context(Arc::new(runtime));

    let result = execute_query_plan(&context, &plan)
        .expect("left-build residual one-shot leaf stream probe should execute through compiled path");
    let batch = result.output().as_host_arrow().expect("compiled output should be host Arrow");
    let metrics = result.metrics();
    let residual_join = residual_join.to_string();

    assert_eq!(batch.num_rows(), 1);
    assert_eq!(int32_triples_sorted(batch, 0, 1, 2), vec![(Some(4), Some(40), Some(45))]);
    assert_eq!(metrics.compiled_pipeline_attempts, 1);
    assert_eq!(metrics.compiled_pipeline_tasks_executed, pipeline.tasks().len() as u64);
    let residual_join_rows = metrics
        .operators
        .iter()
        .filter(|operator| {
            operator.node_id == residual_join
                && operator.join.as_ref().is_some_and(|join| {
                    join.residual_predicate
                        && join.execution_strategy == "bounded_mixed_keyed_residual_join"
                        && join.build_side == JoinSide::Left.to_string()
                        && join.probe_side == JoinSide::Right.to_string()
                })
        })
        .collect::<Vec<_>>();
    assert!(
        !residual_join_rows.is_empty(),
        "operator rows: {:?}; strategy decisions: {:?}",
        metrics.operators,
        metrics.join_strategy_decisions
    );
    assert_eq!(
        residual_join_rows
            .iter()
            .filter_map(|operator| operator.join.as_ref())
            .map(|join| join.probe_rows)
            .sum::<u64>(),
        4
    );
    assert_eq!(
        residual_join_rows
            .iter()
            .filter_map(|operator| operator.join.as_ref())
            .map(|join| join.output_rows)
            .sum::<u64>(),
        1
    );
    assert_eq!(metrics.one_shot_join_count, 0);
    let build_alias_acquires = metrics
        .reservation_events
        .iter()
        .filter(|event| {
            event.node_id.as_deref() == Some(residual_join.as_str())
                && event.memory_boundary == "join_build_side_stream_alias"
                && event.action == MemoryReservationEventAction::AliasAcquire
        })
        .count();
    let build_alias_releases = metrics
        .reservation_events
        .iter()
        .filter(|event| {
            event.node_id.as_deref() == Some(residual_join.as_str())
                && event.memory_boundary == "join_build_side_stream_alias"
                && event.action == MemoryReservationEventAction::AliasRelease
        })
        .count();
    assert!(build_alias_acquires > 0, "residual stream should alias the retained build side");
    assert_eq!(build_alias_acquires, build_alias_releases);
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
    let gpu_inputs = nexus_query_engine::runtime::resident::GpuResidentInputMap::new();
    let mut metrics = QueryMetrics::default();
    let mut executor = CompiledGpuTaskExecutor::new(&context, &plan, &pipeline, &gpu_inputs, &mut metrics).unwrap();

    let error = executor.take_host_output().expect_err("empty host sink should be a structured invariant error");
    let rendered = error.to_string();

    assert_eq!(error.code(), nexus_query_engine::error::QueryErrorCode::InvalidPlan);
    assert_eq!(error.facts().get("compiled_task_id"), Some("t5"));
    assert_eq!(error.facts().get("sink_boundary_kind"), Some("host_arrow_materialization"));
    assert!(!rendered.contains("BoundedJoinExecutionFamily"));
    assert!(!rendered.contains("BoundedJoinExecutionPath"));
    assert!(!rendered.contains("bounded_join_execution_path"));
}
