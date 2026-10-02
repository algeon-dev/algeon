use query_engine::{
    admission::{bounded_join_modeled_facts, bounded_join_unsupported_shape_facts},
    dimension_pruning::local_dimension_pruning_resolver,
    error::QueryErrorCode,
    exec::{
        ExecutionMetrics, QueryExecutionRequest, admit_query_plan, execute_query_plan as engine_execute_query_plan,
    },
    observability::{RuntimeObservationPayload, RuntimeTopologyRecord},
    pipeline::SortStrategy,
    rapids_backend::rapids_analysis_context,
    runtime::resident::GpuResidentInputMap,
};

use super::*;

// `cudf::Runtime` explicitly implements Send and Sync: it re-enters its
// device at each FFI boundary. Clippy cannot see that cross-crate unsafe impl.
#[expect(clippy::arc_with_non_send_sync, reason = "cudf::Runtime explicitly implements Send and Sync")]
fn shared_runtime(runtime: cudf::Runtime) -> Arc<cudf::Runtime> {
    Arc::new(runtime)
}

#[test]
fn compiled_gpu_executor_runs_root_hash_join_build_probe_materialize() {
    let runtime = cudf::Runtime::new().expect("compiled GPU execution test requires a CUDA runtime");
    let dir = unique_temp_dir("query-engine-compiled-root-hash-join");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    write_join_parquets(&left_path, &right_path);

    let compiled_plan = root_join_data_plan(&left_path, &right_path);
    let legacy_plan = root_join_data_plan(&left_path, &right_path);
    let analysis = compiled_plan.capability_analysis(rapids_analysis_context(PlanningPolicy::default()));
    let pipeline = compile_pipeline(&compiled_plan, &analysis).expect("root join should compile");
    assert_eq!(
        compiled_bounded_pipeline_support(&pipeline).expect("support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );
    let scheduler = CompiledPipelineScheduler::new(&pipeline).expect("pipeline should be schedulable");
    let context = bounded_context(shared_runtime(runtime), &compiled_plan);
    let compiled_metrics_recorder = ExecutionMetrics::detached("compiled_metrics");
    let compiled_metrics_context = compiled_metrics_recorder.scope_context(&context).unwrap();
    let gpu_inputs = GpuResidentInputMap::new();

    let (state, compiled_batch) = {
        let mut executor = CompiledGpuTaskExecutor::new(
            &compiled_metrics_context,
            &compiled_plan,
            PlanningPolicy::default(),
            &pipeline,
            &gpu_inputs,
            &compiled_metrics_recorder,
        )
        .unwrap();
        let state = scheduler.run(&mut executor).expect("compiled root hash join should execute");
        let output = executor.take_host_output().expect("compiled host sink should materialize");
        (state, output)
    };
    let compiled_metrics = compiled_metrics_recorder.snapshot();
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
    let runtime = cudf::Runtime::new().expect("compiled GPU execution test requires a CUDA runtime");
    let dir = unique_temp_dir("query-engine-compiled-shared-source");
    let path = dir.join("shared.parquet");
    write_self_join_parquet(&path);

    let plan = shared_source_self_join_data_plan(&path);
    let analysis = plan.capability_analysis(rapids_analysis_context(PlanningPolicy::default()));
    let pipeline = compile_pipeline(&plan, &analysis).expect("shared-source self join should compile");
    assert_task_kind_count(&pipeline, PipelineTaskKind::SourceRead, 1);
    let shared_resource = pipeline
        .resources()
        .iter()
        .find(|resource| resource.consumers().len() == 2)
        .expect("one source resource should feed both join inputs")
        .id();
    assert_eq!(
        compiled_bounded_query_plan_support(&plan, &analysis, &pipeline)
            .expect("plan-aware support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );
    let scheduler = CompiledPipelineScheduler::new(&pipeline).expect("pipeline should be schedulable");
    let context = bounded_context(shared_runtime(runtime), &plan);
    let metrics_recorder = ExecutionMetrics::detached("metrics");
    let metrics_context = metrics_recorder.scope_context(&context).unwrap();
    let gpu_inputs = GpuResidentInputMap::new();

    let (state, compiled_batch) = {
        let mut executor = CompiledGpuTaskExecutor::new(
            &metrics_context,
            &plan,
            PlanningPolicy::default(),
            &pipeline,
            &gpu_inputs,
            &metrics_recorder,
        )
        .unwrap();
        let state = scheduler.run(&mut executor).expect("compiled shared-source join should execute");
        let output = executor.take_host_output().expect("compiled host sink should materialize");
        (state, output)
    };
    let metrics = metrics_recorder.snapshot();

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
fn execute_query_plan_runs_grouped_aggregate_join_build_input_through_admission() {
    let runtime = cudf::Runtime::new().expect("compiled GPU execution test requires a CUDA runtime");
    let dir = unique_temp_dir("query-engine-compiled-aggregate-join-input");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    write_aggregate_join_input_parquets(&left_path, &right_path);

    let optimizer_options = NativeOptimizerOptions {
        join_build_side_selection: JoinBuildSideSelectionOptions::disabled(),
        ..NativeOptimizerOptions::default()
    };
    let plan = optimize_query_plan(
        aggregate_join_build_input_data_plan(&left_path, &right_path),
        &optimizer_options,
        rapids_analysis_context(PlanningPolicy::default()),
        local_dimension_pruning_resolver(),
    )
    .plan;
    assert_eq!(plan.dynamic_filter_routes().len(), 1);
    assert!(matches!(plan.dynamic_filter_routes()[0].producer, DynamicFilterProducer::AggregateGroupKeys { .. }));
    let analysis = plan.capability_analysis(rapids_analysis_context(PlanningPolicy::default()));
    let pipeline = compile_pipeline(&plan, &analysis).expect("aggregate join input plan should compile");
    assert_eq!(
        compiled_bounded_query_plan_support(&plan, &analysis, &pipeline)
            .expect("plan-aware support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );

    let decision = admit(
        &plan,
        StatisticalAggregateExecutionMode::default(),
        PlanningPolicy::default(),
        rapids_analysis_context(PlanningPolicy::default()),
    )
    .expect("bounded admission should not error");
    let admission = decision.admission().expect("a grouped aggregate join-build input is an executable composition");
    assert_eq!(
        admission.pipeline(),
        &pipeline,
        "execution and reporting must share the pipeline the admitted candidate compiled"
    );

    let runtime = shared_runtime(runtime);
    let context = runtime
        .acquire_context_for_plan(&plan)
        .expect("an admitted grouped aggregate join-build input plan needs no envelope recognizer");
    let result = execute_query_plan(&context, &plan).expect("admitted aggregate join input plan should execute");
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(int32_int64_pairs_sorted(batch, 0, 1), vec![(Some(1), Some(15)), (Some(2), Some(20))]);
}

#[test]
fn execute_query_plan_runs_join_sort_limit_filter_through_admission() {
    // Join → Sort → Limit → Filter is decided by the Limit/Unary task contracts,
    // not by the retired post-limit projection allowlist: the filter stays above
    // the limit and still executes natively with the same rows.
    let runtime = cudf::Runtime::new().expect("compiled GPU execution test requires a CUDA runtime");
    let dir = unique_temp_dir("query-engine-compiled-sort-limit-filter");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    write_join_parquets(&left_path, &right_path);

    let plan = root_join_sort_limit_filter_data_plan(&left_path, &right_path);
    let analysis = plan.capability_analysis(rapids_analysis_context(PlanningPolicy::default()));
    let pipeline = compile_pipeline(&plan, &analysis).expect("join sort limit filter plan should compile");
    assert_eq!(
        compiled_bounded_query_plan_support(&plan, &analysis, &pipeline)
            .expect("plan-aware support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );

    let decision = admit(
        &plan,
        StatisticalAggregateExecutionMode::default(),
        PlanningPolicy::default(),
        rapids_analysis_context(PlanningPolicy::default()),
    )
    .expect("bounded admission should not error");
    let admission = decision.admission().expect("join sort limit filter is an executable composition");
    assert_eq!(
        admission.pipeline(),
        &pipeline,
        "execution and reporting must share the pipeline the admitted candidate compiled"
    );

    assert_task_kind_count(&pipeline, PipelineTaskKind::Limit, 1);
    assert_task_kind_count(&pipeline, PipelineTaskKind::Unary, 1);
    let limit_position =
        pipeline.tasks().iter().position(|task| task.kind() == PipelineTaskKind::Limit).expect("one limit task");
    let filter_position =
        pipeline.tasks().iter().position(|task| task.kind() == PipelineTaskKind::Unary).expect("one filter task");
    assert!(
        limit_position < filter_position,
        "the limit must stay below the filter instead of being reordered around it"
    );

    let scheduler = CompiledPipelineScheduler::new(&pipeline).expect("pipeline should be schedulable");
    let context = compiled_executor_context(shared_runtime(runtime), &plan);
    let metrics_recorder = ExecutionMetrics::detached("metrics");
    let metrics_context = metrics_recorder.scope_context(&context).unwrap();
    let gpu_inputs = GpuResidentInputMap::new();
    let (state, batch) = {
        let mut executor = CompiledGpuTaskExecutor::new(
            &metrics_context,
            &plan,
            admission.policy(),
            &pipeline,
            &gpu_inputs,
            &metrics_recorder,
        )
        .unwrap();
        let state = scheduler.run(&mut executor).expect("join sort limit filter should execute");
        let output = executor.take_host_output().expect("compiled host sink should materialize");
        (state, output)
    };

    assert_eq!(state.live_resource_count(), 0);
    // Inner join yields (2, 200) and (4, 400); descending sort plus fetch 1 keeps
    // (4, 400), which also passes the trailing right_value > 100 filter.
    assert_eq!(batch.num_rows(), 1);
    assert_eq!(int32_values(&batch, 0), vec![Some(4)]);
    assert_eq!(int32_values(&batch, 1), vec![Some(400)]);
}

#[test]
fn execute_query_plan_runs_independent_union_branches_through_unified_admission() {
    let runtime = cudf::Runtime::new().expect("compiled GPU execution test requires a CUDA runtime");
    let schema = Arc::new(Schema::new(vec![Field::new("order_key", DataType::Int32, false)]));
    let left_batch =
        RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef])
            .expect("left literal batch");
    let right_batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from(vec![3])) as ArrayRef])
        .expect("right literal batch");
    let mut builder = QueryPlanBuilder::new();
    let left =
        builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(Arc::clone(&schema), vec![left_batch])));
    let left = builder.add_projection(
        left,
        vec![ProjectionOutput::new(Expr::column("order_key"), Field::new("order_key", DataType::Int32, false))],
    );
    let right =
        builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(Arc::clone(&schema), vec![right_batch])));
    let right = builder.add_projection(
        right,
        vec![ProjectionOutput::new(Expr::column("order_key"), Field::new("order_key", DataType::Int32, false))],
    );
    let union = builder.add_union(vec![left, right], Arc::clone(&schema));
    let plan = builder.finish_with_host_sink(union);

    let decision = admit(
        &plan,
        StatisticalAggregateExecutionMode::default(),
        PlanningPolicy::default(),
        rapids_analysis_context(PlanningPolicy::default()),
    )
    .expect("bounded admission should not error");
    let admission = decision.admission().expect("independent union branches use the same compiled admission");
    assert!(
        admission.pipeline().tasks().iter().any(|task| task.kind() == PipelineTaskKind::Union),
        "both branches compile into one pipeline behind a union task"
    );

    let runtime = shared_runtime(runtime);
    let context = runtime.acquire_context_for_plan(&plan).expect("admitted union plan acquires a context");
    let result = execute_query_plan(&context, &plan).expect("admitted union plan should execute width 1");
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    let mut values = int32_values(batch, 0);
    values.sort();
    assert_eq!(values, vec![Some(1), Some(2), Some(3)], "every branch input is computed exactly once");
}

#[test]
fn execute_query_plan_applies_limit_offset_once_through_fused_topk() {
    // ORDER BY right_value DESC LIMIT 1 OFFSET 1 over the join keeps (2, 200):
    // the fused top-k applies skip+fetch, and the limit pass-through drops the
    // marker without applying it again.
    let runtime = cudf::Runtime::new().expect("compiled GPU execution test requires a CUDA runtime");
    let dir = unique_temp_dir("query-engine-compiled-sort-limit-offset");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    write_join_parquets(&left_path, &right_path);

    let plan = root_join_sort_limit_offset_data_plan(&left_path, &right_path);
    let analysis = plan.capability_analysis(rapids_analysis_context(PlanningPolicy::default()));
    let pipeline = compile_pipeline(&plan, &analysis).expect("join sort limit offset plan should compile");
    assert_eq!(
        compiled_bounded_query_plan_support(&plan, &analysis, &pipeline)
            .expect("plan-aware support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );

    let sort_task =
        pipeline.tasks().iter().find(|task| task.kind() == PipelineTaskKind::FinalSort).expect("one sort task");
    assert!(
        matches!(sort_task.sort_strategy(), Some(SortStrategy::FusedTopK { .. })),
        "sort feeding a skip+fetch limit fuses into top-k, got {:?}",
        sort_task.sort_strategy()
    );

    let decision = admit(
        &plan,
        StatisticalAggregateExecutionMode::default(),
        PlanningPolicy::default(),
        rapids_analysis_context(PlanningPolicy::default()),
    )
    .expect("bounded admission should not error");
    let admission = decision.admission().expect("join sort limit offset is an executable composition");
    assert_eq!(
        admission.pipeline(),
        &pipeline,
        "execution and reporting must share the pipeline the admitted candidate compiled"
    );

    let scheduler = CompiledPipelineScheduler::new(&pipeline).expect("pipeline should be schedulable");
    let context = compiled_executor_context(shared_runtime(runtime), &plan);
    let metrics_recorder = ExecutionMetrics::detached("metrics");
    let metrics_context = metrics_recorder.scope_context(&context).unwrap();
    let gpu_inputs = GpuResidentInputMap::new();
    let (state, batch) = {
        let mut executor = CompiledGpuTaskExecutor::new(
            &metrics_context,
            &plan,
            admission.policy(),
            &pipeline,
            &gpu_inputs,
            &metrics_recorder,
        )
        .unwrap();
        let state = scheduler.run(&mut executor).expect("join sort limit offset should execute");
        let output = executor.take_host_output().expect("compiled host sink should materialize");
        (state, output)
    };

    assert_eq!(state.live_resource_count(), 0);
    assert_eq!(batch.num_rows(), 1);
    assert_eq!(int32_values(&batch, 0), vec![Some(2)]);
    assert_eq!(int32_values(&batch, 1), vec![Some(200)]);
}

#[test]
fn compiled_gpu_executor_streams_grouped_aggregate_join_build_input_source() {
    let runtime = cudf::Runtime::new().expect("compiled GPU execution test requires a CUDA runtime");
    let dir = unique_temp_dir("query-engine-compiled-aggregate-join-input-stream");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    write_aggregate_join_input_parquets(&left_path, &right_path);

    let optimizer_options = NativeOptimizerOptions {
        join_build_side_selection: JoinBuildSideSelectionOptions::disabled(),
        ..NativeOptimizerOptions::default()
    };
    let plan = optimize_query_plan(
        aggregate_join_build_input_data_plan(&left_path, &right_path),
        &optimizer_options,
        rapids_analysis_context(PlanningPolicy::default()),
        local_dimension_pruning_resolver(),
    )
    .plan;
    let analysis = plan.capability_analysis(rapids_analysis_context(PlanningPolicy::default()));
    let pipeline = compile_pipeline(&plan, &analysis).expect("aggregate join input plan should compile");
    assert_eq!(
        compiled_bounded_query_plan_support(&plan, &analysis, &pipeline)
            .expect("plan-aware support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );
    let scheduler = CompiledPipelineScheduler::new(&pipeline).expect("pipeline should be schedulable");
    let context = compiled_executor_context(shared_runtime(runtime), &plan);
    let metrics_recorder = ExecutionMetrics::detached("metrics");
    let metrics_context = metrics_recorder.scope_context(&context).unwrap();
    let gpu_inputs = GpuResidentInputMap::new();

    let (state, batch) = {
        let mut executor = CompiledGpuTaskExecutor::new(
            &metrics_context,
            &plan,
            PlanningPolicy::default(),
            &pipeline,
            &gpu_inputs,
            &metrics_recorder,
        )
        .unwrap();
        let state = scheduler.run(&mut executor).expect("compiled aggregate join input should execute");
        let output = executor.take_host_output().expect("compiled host sink should materialize");
        (state, output)
    };
    let metrics = metrics_recorder.snapshot();

    assert_eq!(state.live_resource_count(), 0);
    assert_eq!(int32_int64_pairs_sorted(&batch, 0, 1), vec![(Some(1), Some(15)), (Some(2), Some(20))]);
    assert_eq!(
        metrics.partial_aggregate_count, 2,
        "the aggregate input source has two row-group chunks and should be reduced as chunk-local partials"
    );
    assert_compiled_partial_aggregate_uses_work_reservation_transfer(&metrics);
    assert_compiled_final_merge_uses_partitioned_strategy(&metrics);
    // The group-key snapshot is taken from the merged partition states, so the
    // aggregate that feeds a dynamic-filter route still finalizes lazily.
    assert_eq!(
        metrics
            .aggregate_strategy_decisions
            .iter()
            .filter_map(|decision| decision.facts.get("partitioned_merge_streamed_finalization"))
            .map(String::as_str)
            .collect::<Vec<_>>(),
        vec!["true"],
        "a group-key dynamic-filter producer must not force eager finalization: {:?}",
        metrics.aggregate_strategy_decisions
    );
    let produced = metrics
        .dynamic_filter_runtime_rows
        .iter()
        .filter(|row| row.event == "produced" && row.producer_kind == "aggregate_group_keys")
        .collect::<Vec<_>>();
    assert_eq!(produced.len(), 1, "{:?}", metrics.dynamic_filter_runtime_rows);
    assert_eq!(
        (produced[0].producer_rows, produced[0].payload_min_i64, produced[0].payload_max_i64),
        (Some(3), Some(1), Some(3)),
        "the payload must cover every group key and carry real key values, not unfinalized merge state"
    );
}

#[test]
fn compiled_gpu_executor_streams_shared_source_for_join_build_and_grouped_aggregate() {
    let runtime = cudf::Runtime::new().expect("compiled GPU execution test requires a CUDA runtime");
    let dir = unique_temp_dir("query-engine-compiled-shared-source-aggregate");
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
        rapids_analysis_context(PlanningPolicy::default()),
        local_dimension_pruning_resolver(),
    )
    .plan;
    let analysis = plan.capability_analysis(rapids_analysis_context(PlanningPolicy::default()));
    let pipeline = compile_pipeline(&plan, &analysis).expect("shared-source aggregate plan should compile");
    assert!(
        pipeline.debug_render().contains("kind=shared_resident_producer"),
        "shared source must remain represented as one fan-out resource"
    );
    assert_eq!(
        compiled_bounded_query_plan_support(&plan, &analysis, &pipeline)
            .expect("plan-aware support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );
    let context = bounded_context(shared_runtime(runtime), &plan);
    let result =
        execute_query_plan(&context, &plan).expect("shared source aggregate executes through the engine boundary");
    let batch = result.output().as_host_arrow().expect("shared source host sink materializes");
    let metrics = result.metrics();

    assert_eq!(
        int32_int64_pairs_sorted(batch, 0, 1),
        vec![(Some(1), Some(15)), (Some(1), Some(15)), (Some(2), Some(20))]
    );
    assert_eq!(
        metrics.partial_aggregate_count, 2,
        "the shared aggregate branch should clone the deferred source stream and reduce both row-group chunks"
    );
    assert_compiled_final_merge_uses_partitioned_strategy(metrics);
}

#[test]
fn execute_query_plan_routes_nonlinear_topk_and_subquery_filter_paths_through_compiled_pipeline() {
    let runtime = cudf::Runtime::new().expect("compiled GPU execution test requires a CUDA runtime");
    let runtime = shared_runtime(runtime);
    write_nonlinear_topk_contract_parquets();

    for (label, plan) in [
        (
            "aggregate-over-join-subplan TopK",
            aggregate_over_join_subplan_topk_plan_inner(AggregateOverJoinSubplanTopKVariant::Admitted, true),
        ),
        (
            "resident grouped-aggregate TopK",
            resident_grouped_aggregate_input_post_join_aggregate_topk_plan_inner(
                ResidentGroupedAggregateInputPostJoinAggregateTopKVariant::Admitted,
                true,
            ),
        ),
        ("resident grouped-aggregate subquery filter", resident_grouped_aggregate_subquery_filter_plan(true)),
    ] {
        let analysis = plan.capability_analysis(rapids_analysis_context(PlanningPolicy::default()));
        let pipeline =
            compile_pipeline(&plan, &analysis).unwrap_or_else(|error| panic!("{label}: compile failed: {error}"));
        assert_eq!(
            compiled_bounded_query_plan_support(&plan, &analysis, &pipeline)
                .expect("plan-aware support check should not fail"),
            CompiledBoundedPipelineSupport::Supported,
            "{label}: non-linear plan should be admitted by compiled support"
        );
        let context = bounded_context(Arc::clone(&runtime), &plan);

        let result = execute_query_plan(&context, &plan)
            .unwrap_or_else(|error| panic!("{label}: compiled execution failed: {error}"));
        let batch = result.output().as_host_arrow().expect("compiled output should be host Arrow");
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

    remove_nonlinear_topk_contract_parquets();
}

#[test]
fn execute_query_plan_routes_root_hash_join_through_compiled_pipeline() {
    let runtime = cudf::Runtime::new().expect("compiled GPU execution test requires a CUDA runtime");
    let dir = unique_temp_dir("query-engine-compiled-root-join");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    write_join_parquets(&left_path, &right_path);

    let plan = root_join_data_plan(&left_path, &right_path);
    let analysis = plan.capability_analysis(rapids_analysis_context(PlanningPolicy::default()));
    let pipeline = compile_pipeline(&plan, &analysis).expect("root join should compile");
    let context = bounded_context(shared_runtime(runtime), &plan);

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
fn observed_compiled_root_join_records_registered_tasks_without_reservation_events() {
    let runtime = cudf::Runtime::new().expect("compiled GPU execution test requires a CUDA runtime");
    let dir = unique_temp_dir("query-engine-observed-compiled-root");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    write_join_parquets(&left_path, &right_path);
    let plan = root_join_data_plan(&left_path, &right_path);
    let pipeline =
        compile_pipeline(&plan, &plan.capability_analysis(rapids_analysis_context(PlanningPolicy::default())))
            .expect("observed root join compiles");
    let (context, observation) = bounded_observed_context(shared_runtime(runtime), &plan);

    let admitted = admit_query_plan(&plan, context.statistical_aggregate_execution_mode(), context.planning_policy())
        .expect("observed root join admits");
    let outcome = engine_execute_query_plan(&context, QueryExecutionRequest::admitted(&admitted));
    let (result, metrics, plan_observation) = outcome.into_parts();
    let metrics = metrics.snapshot();
    assert!(matches!(result, Ok(QueryOutput::HostArrow(_))));
    assert!(plan_observation.is_none(), "one plan must not seal its query observation");
    drop(context);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    let snapshot = loop {
        if let Some(snapshot) = observation.sealed_snapshot() {
            break snapshot;
        }
        assert!(std::time::Instant::now() < deadline, "query observation seals after cleanup");
        std::thread::yield_now();
    };
    assert!(snapshot.events().iter().all(|event| !matches!(event.payload, RuntimeObservationPayload::Reservation(_))));
    assert!(metrics.task_runtime_metrics.values().all(|task| task.operator_kind != "unknown"));
    assert!(snapshot.topology().any(|record| { matches!(record, RuntimeTopologyRecord::TaskDefinition(_)) }));
    let tasks = snapshot
        .topology()
        .filter_map(|record| match record {
            RuntimeTopologyRecord::TaskDefinition(row) => Some((row.plan_execution_id.clone(), row.task_id.clone())),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(tasks.len(), pipeline.tasks().len());
    assert_eq!(tasks.iter().cloned().collect::<std::collections::BTreeSet<_>>().len(), tasks.len());

    let barriers = snapshot
        .topology()
        .filter_map(|record| match record {
            RuntimeTopologyRecord::BarrierDefinition(row) => {
                Some((row.plan_execution_id.clone(), row.barrier_id.to_string()))
            },
            _ => None,
        })
        .collect::<std::collections::BTreeSet<_>>();
    let sinks = snapshot
        .topology()
        .filter_map(|record| match record {
            RuntimeTopologyRecord::SinkDefinition(row) => {
                Some((row.plan_execution_id.clone(), row.sink_id.to_string()))
            },
            _ => None,
        })
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(barriers.len(), pipeline.barriers().len());
    assert_eq!(sinks.len(), 1);
}

#[test]
fn execute_query_plan_routes_keyless_root_join_strategy_through_compiled_pipeline() {
    let runtime = cudf::Runtime::new().expect("compiled GPU execution test requires a CUDA runtime");
    let dir = unique_temp_dir("query-engine-compiled-keyless-root-join");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    write_join_parquets(&left_path, &right_path);
    let plan = keyless_root_join_data_plan(&left_path, &right_path);
    assert!(bounded_pipeline_admission(&plan).unwrap().is_some(), "keyless root join should be admitted");
    let analysis = plan.capability_analysis(rapids_analysis_context(PlanningPolicy::default()));
    let pipeline = compile_pipeline(&plan, &analysis).expect("keyless root join should compile");
    assert_eq!(
        compiled_bounded_query_plan_support(&plan, &analysis, &pipeline)
            .expect("plan-aware support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );
    let context = bounded_context(shared_runtime(runtime), &plan);

    let result = execute_query_plan(&context, &plan).expect("keyless root join should execute");
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
    let runtime = cudf::Runtime::new().expect("compiled GPU execution test requires a CUDA runtime");
    let dir = unique_temp_dir("query-engine-compiled-unary-root-join");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    write_join_parquets(&left_path, &right_path);

    let plan = root_join_with_unary_inputs_data_plan(&left_path, &right_path);
    let analysis = plan.capability_analysis(rapids_analysis_context(PlanningPolicy::default()));
    let pipeline = compile_pipeline(&plan, &analysis).expect("root join with unary inputs should compile");
    assert_eq!(
        compiled_bounded_pipeline_support(&pipeline).expect("support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );
    let context = bounded_context(shared_runtime(runtime), &plan);

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
fn execute_query_plan_routes_aggregate_sort_limit_through_compiled_pipeline() {
    let runtime = cudf::Runtime::new().expect("compiled GPU execution test requires a CUDA runtime");
    let dir = unique_temp_dir("query-engine-compiled-fallback");
    let path = dir.join("input.parquet");
    let source_schema = write_source_unary_parquet(&path);
    let plan = aggregate_sort_limit_data_plan(&path, source_schema);
    let admission = bounded_pipeline_admission(&plan)
        .expect("aggregate admission should not fail")
        .expect("aggregate/sort/limit should use the compiled pipeline");
    let context = bounded_context(shared_runtime(runtime), &plan);

    let result = execute_query_plan(&context, &plan).expect("compiled aggregate/sort/limit should execute");
    let batch = result.output().as_host_arrow().expect("host sink should return host Arrow");
    let metrics = result.metrics();

    assert_eq!(batch.num_rows(), 4);
    assert_eq!(metrics.compiled_pipeline_attempts, 1);
    assert_eq!(metrics.compiled_pipeline_tasks_executed, admission.pipeline().tasks().len() as u64);
    assert_eq!(metrics.compiled_pipeline_host_arrow_sinks, 1);
}

#[test]
fn execute_query_plan_routes_gpu_resident_sink_join_through_compiled_pipeline() {
    let runtime = cudf::Runtime::new().expect("compiled GPU execution test requires a CUDA runtime");
    let dir = unique_temp_dir("query-engine-compiled-gpu-sink");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    write_join_parquets(&left_path, &right_path);
    let plan = root_join_gpu_sink_data_plan(&left_path, &right_path);
    let decision = admit(
        &plan,
        StatisticalAggregateExecutionMode::default(),
        PlanningPolicy::default(),
        rapids_analysis_context(PlanningPolicy::default()),
    )
    .expect("bounded admission should not fail");
    let admission = decision.admission().expect("GPU sink root join should be admitted");
    assert_eq!(admission.pipeline().sink().kind(), SinkBoundaryKind::GpuResidentOutput);
    assert!(
        bounded_pipeline_rejection_projection(&plan).expect("bounded rejection projection should not fail").is_none(),
        "GPU sink should stay inside the compiled bounded envelope"
    );
    assert!(
        bounded_join_modeled_facts(&plan, &decision, rapids_analysis_context(PlanningPolicy::default())).is_some(),
        "GPU sink plans admitted by compiled execution must emit modeled bounded-join facts"
    );
    assert!(
        bounded_join_unsupported_shape_facts(&plan, &decision, rapids_analysis_context(PlanningPolicy::default()))
            .is_none(),
        "GPU sink plans admitted by compiled execution must not emit unsupported bounded-join facts"
    );
    let mut passthrough_builder = QueryPlanBuilder::new();
    let passthrough_source = passthrough_builder.add_source(SourceNode::GpuResident(GpuResidentSource::new(
        "compiled_root_join",
        Arc::clone(plan.output_schema()),
    )));
    let passthrough_plan = passthrough_builder.finish_with_host_sink(passthrough_source);
    let root_evidence = AttemptCapabilityEvidence::for_query_plan(&plan);
    let passthrough_evidence = AttemptCapabilityEvidence::for_query_plan(&passthrough_plan);
    let composed_evidence = AttemptCapabilityEvidence::aggregate([root_evidence, passthrough_evidence])
        .expect("test evidence has one manifest");
    let context = bounded_context_for_evidence_items(shared_runtime(runtime), [composed_evidence]);
    let result = execute_query_plan(&context, &plan).expect("compiled GPU sink join should execute");
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
fn pipeline_validation_rejects_host_sink_output_lifetime() {
    let source = PipelineTaskId::new(0);
    let output = ResidentResourceId::new(0);
    let pipeline =
        single_sink_output_pipeline(SinkBoundaryKind::HostArrowMaterialization, source, vec![(source, output)]);

    let error = pipeline.validate().expect_err("host sinks cannot retain sink-output resources");

    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
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

    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
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

    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
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
    CompiledPipeline::from_parts_unvalidated(
        tasks,
        vec![],
        vec![
            PipelineBarrier::new(PipelineBarrierId::new(0), PipelineBarrierKind::SinkBoundary)
                .with_upstream_tasks(task_ids.into_iter().collect()),
        ],
        resources,
        SinkBoundary::new(kind, sink_input),
    )
}

#[test]
fn execute_query_plan_routes_root_hash_join_aggregate_through_compiled_pipeline() {
    let runtime = cudf::Runtime::new().expect("compiled GPU execution test requires a CUDA runtime");
    let dir = unique_temp_dir("query-engine-compiled-single-join-aggregate");
    let customers_path = dir.join("customers.parquet");
    let orders_path = dir.join("orders.parquet");
    write_single_join_post_root_parquets(&customers_path, &orders_path);
    let plan = single_join_post_root_aggregate_data_plan(&customers_path, &orders_path);
    let analysis = plan.capability_analysis(rapids_analysis_context(PlanningPolicy::default()));
    let pipeline = compile_pipeline(&plan, &analysis).expect("single-join aggregate plan should compile");
    assert_task_kind_count(&pipeline, PipelineTaskKind::Aggregate, 1);
    assert_task_kind_count(&pipeline, PipelineTaskKind::FinalSort, 0);
    assert_task_kind_count(&pipeline, PipelineTaskKind::Limit, 0);
    assert_eq!(
        compiled_bounded_pipeline_support(&pipeline).expect("support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );
    let context = bounded_context(shared_runtime(runtime), &plan);

    let result = execute_query_plan(&context, &plan).expect("aggregate graph should execute through compiled path");
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
    let runtime = cudf::Runtime::new().expect("compiled GPU execution test requires a CUDA runtime");
    let dir = unique_temp_dir("query-engine-compiled-string-group-aggregate");
    let customers_path = dir.join("customers.parquet");
    let orders_path = dir.join("orders.parquet");
    write_string_group_join_post_root_parquets(&customers_path, &orders_path);
    let plan = string_group_join_post_root_aggregate_data_plan(&customers_path, &orders_path);
    let analysis = plan.capability_analysis(rapids_analysis_context(PlanningPolicy::default()));
    let pipeline = compile_pipeline(&plan, &analysis).expect("string grouped aggregate plan should compile");
    assert_task_kind_count(&pipeline, PipelineTaskKind::Aggregate, 1);
    assert_eq!(
        compiled_bounded_query_plan_support(&plan, &analysis, &pipeline)
            .expect("plan-aware support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );
    let context = bounded_context(shared_runtime(runtime), &plan);

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
}

fn assert_compiled_partial_aggregate_uses_work_reservation_transfer(metrics: &QueryMetrics) {
    assert!(metrics.partial_aggregate_count > 0);
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
}

#[test]
fn execute_query_plan_routes_root_hash_join_final_sort_limit_through_compiled_pipeline() {
    let runtime = cudf::Runtime::new().expect("compiled GPU execution test requires a CUDA runtime");
    let dir = unique_temp_dir("query-engine-compiled-root-sort-limit");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    write_join_parquets(&left_path, &right_path);
    let plan = root_join_sort_limit_data_plan(&left_path, &right_path);
    let analysis = plan.capability_analysis(rapids_analysis_context(PlanningPolicy::default()));
    let pipeline = compile_pipeline(&plan, &analysis).expect("root join final-sort/limit plan should compile");
    assert_task_kind_count(&pipeline, PipelineTaskKind::Aggregate, 0);
    assert_task_kind_count(&pipeline, PipelineTaskKind::FinalSort, 1);
    assert_task_kind_count(&pipeline, PipelineTaskKind::Limit, 1);
    assert_eq!(
        compiled_bounded_pipeline_support(&pipeline).expect("support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );
    let context = bounded_context(shared_runtime(runtime), &plan);

    let result = execute_query_plan(&context, &plan).expect("final-sort/limit graph should execute");
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
fn execute_query_plan_routes_single_join_post_root_graph_through_compiled_pipeline() {
    let runtime = cudf::Runtime::new().expect("compiled GPU execution test requires a CUDA runtime");
    let dir = unique_temp_dir("query-engine-compiled-single-join-post-root");
    let customers_path = dir.join("customers.parquet");
    let orders_path = dir.join("orders.parquet");
    write_single_join_post_root_parquets(&customers_path, &orders_path);
    let plan = single_join_post_root_aggregate_sort_limit_data_plan(&customers_path, &orders_path);
    let analysis = plan.capability_analysis(rapids_analysis_context(PlanningPolicy::default()));
    let pipeline = compile_pipeline(&plan, &analysis).expect("single-join post-root plan should compile");
    assert_eq!(
        compiled_bounded_pipeline_support(&pipeline).expect("support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );
    let context = bounded_context(shared_runtime(runtime), &plan);

    let result =
        execute_query_plan(&context, &plan).expect("single-join post-root graph should execute through compiled path");
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
    let runtime = cudf::Runtime::new().expect("compiled GPU execution test requires a CUDA runtime");
    let dir = unique_temp_dir("query-engine-compiled-multi-join");
    let a_path = dir.join("a.parquet");
    let b_path = dir.join("b.parquet");
    let c_path = dir.join("c.parquet");
    write_multi_join_parquets(&a_path, &b_path, &c_path);
    let plan = simple_multi_join_plan(&a_path, &b_path, &c_path);
    let analysis = plan.capability_analysis(rapids_analysis_context(PlanningPolicy::default()));
    let pipeline = compile_pipeline(&plan, &analysis).expect("simple multi-join plan should compile");
    assert_eq!(
        compiled_bounded_pipeline_support(&pipeline).expect("support check should not fail"),
        CompiledBoundedPipelineSupport::Supported
    );
    let context = bounded_context(shared_runtime(runtime), &plan);

    let result = execute_query_plan(&context, &plan).expect("multi-join graph should execute");
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
    let runtime = cudf::Runtime::new().expect("compiled GPU execution test requires a CUDA runtime");
    let dir = unique_temp_dir("query-engine-compiled-residual-stream");
    let a_path = dir.join("a.parquet");
    let b_path = dir.join("b.parquet");
    let c_path = dir.join("c.parquet");
    write_residual_stream_join_parquets(&a_path, &b_path, &c_path);
    let plan = residual_one_shot_frame_stream_multi_join_plan(&a_path, &b_path, &c_path);
    let analysis = plan.capability_analysis(rapids_analysis_context(PlanningPolicy::default()));
    let pipeline = compile_pipeline(&plan, &analysis).expect("residual stream multi-join plan should compile");
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
        .expect("fixture should contain the downstream residual join");
    let context = bounded_context(shared_runtime(runtime), &plan);

    let result = execute_query_plan(&context, &plan)
        .expect("residual one-shot stream probe should execute through compiled path");
    let batch = result.output().as_host_arrow().expect("compiled output should be host Arrow");
    let metrics = result.metrics();
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
}

#[test]
fn execute_query_plan_streams_residual_one_shot_probe_from_leaf_local_source() {
    let runtime = cudf::Runtime::new().expect("compiled GPU execution test requires a CUDA runtime");
    let dir = unique_temp_dir("query-engine-compiled-residual-leaf-stream");
    let a_path = dir.join("a.parquet");
    let b_path = dir.join("b.parquet");
    let c_path = dir.join("c.parquet");
    write_residual_stream_join_parquets(&a_path, &b_path, &c_path);
    let plan = residual_one_shot_leaf_stream_join_plan(&a_path, &c_path);
    let analysis = plan.capability_analysis(rapids_analysis_context(PlanningPolicy::default()));
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
    let context = bounded_context(shared_runtime(runtime), &plan);

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
}

#[test]
fn execute_query_plan_streams_residual_one_shot_right_probe_with_left_build() {
    let runtime = cudf::Runtime::new().expect("compiled GPU execution test requires a CUDA runtime");
    let dir = unique_temp_dir("query-engine-compiled-residual-left-build-stream");
    let a_path = dir.join("a.parquet");
    let b_path = dir.join("b.parquet");
    let c_path = dir.join("c.parquet");
    write_residual_stream_join_parquets(&a_path, &b_path, &c_path);
    let plan = residual_one_shot_leaf_stream_left_build_join_plan(&a_path, &c_path);
    let analysis = plan.capability_analysis(rapids_analysis_context(PlanningPolicy::default()));
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
    let context = bounded_context(shared_runtime(runtime), &plan);

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
}

#[test]
fn compiled_host_sink_diagnostics_are_structured_without_legacy_routing_terms() {
    let runtime = cudf::Runtime::new().expect("compiled GPU execution test requires a CUDA runtime");
    let dir = unique_temp_dir("query-engine-compiled-host-sink-diagnostics");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    write_join_parquets(&left_path, &right_path);
    let plan = root_join_data_plan(&left_path, &right_path);
    let analysis = plan.capability_analysis(rapids_analysis_context(PlanningPolicy::default()));
    let pipeline = compile_pipeline(&plan, &analysis).expect("root join should compile");
    let context = bounded_context(shared_runtime(runtime), &plan);
    let gpu_inputs = GpuResidentInputMap::new();
    let metrics_recorder = ExecutionMetrics::detached("metrics");
    let metrics_context = metrics_recorder.scope_context(&context).unwrap();
    let mut executor = CompiledGpuTaskExecutor::new(
        &metrics_context,
        &plan,
        PlanningPolicy::default(),
        &pipeline,
        &gpu_inputs,
        &metrics_recorder,
    )
    .unwrap();

    let error = executor.take_host_output().expect_err("empty host sink should be a structured invariant error");
    let rendered = error.to_string();

    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.facts().get("compiled_task_id"), Some("t5"));
    assert_eq!(error.facts().get("sink_boundary_kind"), Some("host_arrow_materialization"));
    assert!(!rendered.contains("BoundedJoinExecutionFamily"));
    assert!(!rendered.contains("BoundedJoinExecutionPath"));
    assert!(!rendered.contains("bounded_join_execution_path"));
}
