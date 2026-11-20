use arrow::array::Float64Array;
#[cfg(feature = "cugraph")]
use cudf::testing::device_copy::device_to_device_copy_count_for_test;
use cudf::{DataFrame as CudfDataFrame, testing::run_named_cases};
#[cfg(feature = "cugraph")]
use query_planning::plan::{
    GraphAlgorithm, GraphAlgorithmParams, GraphAlgorithmSpec, GraphBetweennessCentralityParams, GraphBetweennessSeeds,
    GraphConstructionPolicy, GraphPageRankParams, graph_result_schema_for_vertex_type,
};
use query_planning::{
    admission::{bounded_join_modeled_facts, bounded_join_unsupported_shape_facts},
    error::QueryErrorCode,
    pipeline::SortStrategy,
    plan::{
        EdgeDedupStrategy, EdgeNormalizeSpec, EdgeWeightCombiner, GraphConstructionOptions, GraphEdgeColumnBindings,
    },
};
use query_runtime::{
    observability::{RuntimeObservationPayload, RuntimeTopologyRecord},
    runtime::resident::{GpuDataFrame, GpuResidentInputMap, GpuResultHandle, ProducerExecutionIdentity},
};

use super::*;
#[cfg(feature = "cugraph")]
use crate::exec::graph_algorithm::{
    bounded_betweenness::validate_betweenness_seed_policy_for_vertex_count, narrow_id_columns,
    statistics::runtime_graph_statistics_probe,
};
use crate::{
    dimension_pruning::local_dimension_pruning_resolver,
    exec::{
        ExecutionMetrics, QueryExecutionRequest, admit_query_plan,
        bounded::compiled::{CompiledGpuResource, scheduler::CompiledTaskExecutor, state::CompiledPipelineState},
        execute_query_plan as engine_execute_query_plan,
    },
    rapids_backend::rapids_analysis_context,
};

fn compiled_gpu_executor_runs_root_hash_join_build_probe_materialize() {
    let runtime = suite_backend::Runtime::new().expect("compiled GPU execution test requires a CUDA runtime");
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
    let production_metrics = production_result.metrics();
    assert_eq!(production_metrics.compiled_pipeline_attempts, 1);
    assert_eq!(production_metrics.compiled_pipeline_tasks_executed, pipeline.tasks().len() as u64);
    assert_eq!(production_metrics.compiled_pipeline_host_arrow_sinks, 1);
    assert_eq!(production_metrics.hash_join_build_count, 1);
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

fn compiled_gpu_executor_runs_shared_source_hash_join_without_rereading() {
    let runtime = suite_backend::Runtime::new().expect("compiled GPU execution test requires a CUDA runtime");
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
    let runtime = suite_backend::Runtime::new().expect("compiled GPU execution test requires a CUDA runtime");
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
    .unwrap()
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

fn execute_query_plan_runs_join_sort_limit_filter_through_admission() {
    // Join → Sort → Limit → Filter is decided by the Limit/Unary task contracts,
    // not by the retired post-limit projection allowlist: the filter stays above
    // the limit and still executes natively with the same rows.
    let runtime = suite_backend::Runtime::new().expect("compiled GPU execution test requires a CUDA runtime");
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

fn execute_query_plan_runs_independent_union_branches_through_unified_admission() {
    let runtime = suite_backend::Runtime::new().expect("compiled GPU execution test requires a CUDA runtime");
    let schema = Arc::new(Schema::new(vec![Field::new("order_key", DataType::Int32, false)]));
    let left_batch =
        RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef])
            .expect("left literal batch");
    let right_batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from(vec![3])) as ArrayRef])
        .expect("right literal batch");
    let mut builder = QueryPlanBuilder::new();
    let left =
        builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(Arc::clone(&schema), vec![left_batch])));
    let left = builder
        .add_projection(
            left,
            vec![ProjectionOutput::new(Expr::column("order_key"), Field::new("order_key", DataType::Int32, false))],
        )
        .unwrap();
    let right =
        builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(Arc::clone(&schema), vec![right_batch])));
    let right = builder
        .add_projection(
            right,
            vec![ProjectionOutput::new(Expr::column("order_key"), Field::new("order_key", DataType::Int32, false))],
        )
        .unwrap();
    let union = builder.add_union(vec![left, right], Arc::clone(&schema)).unwrap();
    let plan = builder.finish_with_host_sink(union).unwrap();

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

/// Identical UNION ALL branches intern to one producer that the union reads
/// through two inputs. Both reads take the whole frame, so each row appears
/// twice.
fn execute_query_plan_runs_a_union_that_reads_one_producer_twice() {
    let runtime = suite_backend::Runtime::new().expect("compiled GPU execution test requires a CUDA runtime");
    let schema = Arc::new(Schema::new(vec![Field::new("order_key", DataType::Int32, false)]));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from(vec![1, 2, 3])) as ArrayRef])
        .expect("literal batch");
    let runtime = shared_runtime(runtime);
    for computed in [false, true] {
        let mut builder = QueryPlanBuilder::new();
        let mut branch = builder
            .add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(Arc::clone(&schema), vec![batch.clone()])));
        if computed {
            branch = builder
                .add_projection(
                    branch,
                    vec![ProjectionOutput::new(
                        Expr::column("order_key"),
                        Field::new("order_key", DataType::Int32, false),
                    )],
                )
                .unwrap();
        }
        let union = builder.add_union(vec![branch, branch], Arc::clone(&schema)).unwrap();
        let plan = builder.finish_with_host_sink(union).unwrap();

        let context = runtime.acquire_context_for_plan(&plan).expect("union plan acquires a context");
        let result = execute_query_plan(&context, &plan)
            .unwrap_or_else(|error| panic!("computed={computed}: a union of one producer should execute: {error}"));
        let QueryOutput::HostArrow(output) = result.output() else {
            panic!("host sink should materialize one RecordBatch");
        };
        let mut values = int32_values(output, 0);
        values.sort();
        assert_eq!(values, vec![Some(1), Some(1), Some(2), Some(2), Some(3), Some(3)], "computed={computed}");
    }
}

fn execute_query_plan_applies_limit_offset_once_through_fused_topk() {
    // ORDER BY right_value DESC LIMIT 1 OFFSET 1 over the join keeps (2, 200):
    // the fused top-k applies skip+fetch, and the limit pass-through drops the
    // marker without applying it again.
    let runtime = suite_backend::Runtime::new().expect("compiled GPU execution test requires a CUDA runtime");
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

fn compiled_gpu_executor_streams_grouped_aggregate_join_build_input_source() {
    let runtime = suite_backend::Runtime::new().expect("compiled GPU execution test requires a CUDA runtime");
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
    .unwrap()
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

fn compiled_gpu_executor_streams_shared_source_for_join_build_and_grouped_aggregate() {
    let runtime = suite_backend::Runtime::new().expect("compiled GPU execution test requires a CUDA runtime");
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
    .unwrap()
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
    let fact = pipeline
        .resources()
        .iter()
        .find(|resource| resource.consumers().len() == 2)
        .expect("the fact source feeds the join build and the aggregate");
    assert_eq!(
        (fact.data_form(), fact.sharing()),
        (ResidentDataForm::SourceStream, ResourceSharing::IndependentCursorStream)
    );
    let fact_read = pipeline
        .tasks()
        .iter()
        .find(|task| task.id() == fact.producer())
        .expect("the fact source task exists")
        .physical();
    let no_stages = AggregateInputHints { adjacent_stages: false, grouped_string_key: false };
    assert_eq!(
        fact_read,
        &TaskPhysical::SourceRead(SourceReadPlan {
            output: SourceOutput::SharedJoinAggregateStream { adjacent_stages: false, chunk_hints: no_stages },
            dynamic_filter_routes: false,
            prefetch_eligible: false,
        }),
        "a source shared by a join build and an aggregate reads one stream shaped for both"
    );
    assert_eq!(
        source_row_groups_read(&plan, metrics, &fact_path),
        4,
        "each consumer replays the fact source's two row groups through its own cursor, as declared"
    );
    assert_compiled_final_merge_uses_partitioned_strategy(metrics);
}

/// Row groups read during one run by the plan's source of the Parquet file at
/// `path`.
fn source_row_groups_read(plan: &QueryPlan, metrics: &QueryMetrics, path: &Path) -> u64 {
    let source = plan
        .nodes()
        .iter()
        .find(|node| match &node.kind {
            PlanNode::Source(source) => match source.as_ref() {
                SourceNode::LocalParquet(source) => source.files.iter().any(|file| file.path == path),
                _ => false,
            },
            _ => false,
        })
        .expect("the plan reads the Parquet file")
        .id
        .to_string();
    metrics
        .source_read_runtime_rows
        .iter()
        .filter(|row| row.source_node_id == source)
        .map(|row| row.row_group_count)
        .sum()
}

fn compiled_gpu_executor_reads_a_filtered_source_shared_by_two_joins_once() {
    let runtime = suite_backend::Runtime::new().expect("compiled GPU execution test requires a CUDA runtime");
    let dir = unique_temp_dir("query-engine-compiled-shared-filtered-source");
    let dimension_path = dir.join("dimension.parquet");
    let fact_paths = [dir.join("fact_a.parquet"), dir.join("fact_b.parquet")];
    write_shared_filtered_dimension_parquets(&dimension_path, [&fact_paths[0], &fact_paths[1]]);

    let plan = shared_filtered_dimension_fan_out_plan(&dimension_path, [&fact_paths[0], &fact_paths[1]]);
    let analysis = plan.capability_analysis(rapids_analysis_context(PlanningPolicy::default()));
    let pipeline = compile_pipeline(&plan, &analysis).expect("shared filtered dimension plan should compile");
    let filtered = pipeline
        .resources()
        .iter()
        .find(|resource| resource.consumers().len() == 2)
        .expect("the filtered dimension feeds both joins");
    assert_eq!(
        (filtered.data_form(), filtered.sharing()),
        (ResidentDataForm::Frame, ResourceSharing::SharedImmutableFrame),
        "two join builds share one materialized copy of the filtered dimension"
    );
    let context = bounded_context(shared_runtime(runtime), &plan);
    let result = execute_query_plan(&context, &plan).expect("shared filtered dimension plan should execute");
    let batch = result.output().as_host_arrow().expect("shared filtered dimension host sink materializes");

    assert_eq!(int32_int64_pairs_sorted(batch, 0, 1), vec![(Some(1), Some(10)), (Some(3), Some(30))]);
    assert_eq!(
        source_row_groups_read(&plan, result.metrics(), &dimension_path),
        2,
        "the filter reads the dimension's two row groups once for both joins"
    );
}

fn execute_query_plan_routes_nonlinear_topk_and_subquery_filter_paths_through_compiled_pipeline() {
    let runtime = suite_backend::Runtime::new().expect("compiled GPU execution test requires a CUDA runtime");
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
fn gpu_execution_cases_1() {
    run_named_cases(&[
        (
            "exec::bounded::compiled::execution_tests::gpu_execution::execute_query_plan_routes_nonlinear_topk_and_subquery_filter_paths_through_compiled_pipeline",
            execute_query_plan_routes_nonlinear_topk_and_subquery_filter_paths_through_compiled_pipeline,
        ),
        (
            "exec::bounded::compiled::execution_tests::gpu_execution::compiled_gpu_executor_streams_grouped_aggregate_join_build_input_source",
            compiled_gpu_executor_streams_grouped_aggregate_join_build_input_source,
        ),
        (
            "exec::bounded::compiled::execution_tests::gpu_execution::execute_query_plan_sizes_compiled_stream_aggregate_string_group_keys_inside_reservations",
            execute_query_plan_sizes_compiled_stream_aggregate_string_group_keys_inside_reservations,
        ),
        (
            "exec::bounded::compiled::execution_tests::gpu_execution::execute_query_plan_routes_root_hash_join_aggregate_through_compiled_pipeline",
            execute_query_plan_routes_root_hash_join_aggregate_through_compiled_pipeline,
        ),
        (
            "exec::bounded::compiled::execution_tests::gpu_execution::execute_query_plan_runs_join_sort_limit_filter_through_admission",
            execute_query_plan_runs_join_sort_limit_filter_through_admission,
        ),
        (
            "exec::bounded::compiled::execution_tests::gpu_execution::compiled_gpu_executor_reads_a_filtered_source_shared_by_two_joins_once",
            compiled_gpu_executor_reads_a_filtered_source_shared_by_two_joins_once,
        ),
    ]);
}

fn observed_compiled_root_join_records_registered_tasks_without_reservation_events() {
    let runtime = suite_backend::Runtime::new().expect("compiled GPU execution test requires a CUDA runtime");
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
    let runtime = suite_backend::Runtime::new().expect("compiled GPU execution test requires a CUDA runtime");
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

fn execute_query_plan_routes_root_hash_join_with_unary_inputs_through_compiled_pipeline() {
    let runtime = suite_backend::Runtime::new().expect("compiled GPU execution test requires a CUDA runtime");
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
fn gpu_execution_cases_2() {
    run_named_cases(&[
        (
            "exec::bounded::compiled::execution_tests::gpu_execution::execute_query_plan_routes_root_hash_join_with_unary_inputs_through_compiled_pipeline",
            execute_query_plan_routes_root_hash_join_with_unary_inputs_through_compiled_pipeline,
        ),
        (
            "exec::bounded::compiled::execution_tests::gpu_execution::execute_query_plan_routes_gpu_resident_sink_join_through_compiled_pipeline",
            execute_query_plan_routes_gpu_resident_sink_join_through_compiled_pipeline,
        ),
        (
            "exec::bounded::compiled::execution_tests::gpu_execution::compiled_gpu_executor_runs_root_hash_join_build_probe_materialize",
            compiled_gpu_executor_runs_root_hash_join_build_probe_materialize,
        ),
        (
            "exec::bounded::compiled::execution_tests::gpu_execution::compiled_gpu_executor_streams_shared_source_for_join_build_and_grouped_aggregate",
            compiled_gpu_executor_streams_shared_source_for_join_build_and_grouped_aggregate,
        ),
        (
            "exec::bounded::compiled::execution_tests::gpu_execution::compiled_gpu_executor_runs_shared_source_hash_join_without_rereading",
            compiled_gpu_executor_runs_shared_source_hash_join_without_rereading,
        ),
        (
            "exec::bounded::compiled::execution_tests::gpu_execution::execute_query_plan_streams_residual_one_shot_probe_from_leaf_local_source",
            execute_query_plan_streams_residual_one_shot_probe_from_leaf_local_source,
        ),
    ]);
}

#[test]
fn execute_query_plan_routes_aggregate_sort_limit_through_compiled_pipeline() {
    let runtime = suite_backend::Runtime::new().expect("compiled GPU execution test requires a CUDA runtime");
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

fn execute_query_plan_routes_gpu_resident_sink_join_through_compiled_pipeline() {
    let runtime = suite_backend::Runtime::new().expect("compiled GPU execution test requires a CUDA runtime");
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
    let passthrough_plan = passthrough_builder.finish_with_host_sink(passthrough_source).unwrap();
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
    let repeated = execute_query_plan_with_gpu_inputs(&context, &passthrough_plan, &passthrough_inputs)
        .expect("borrowed inputs remain reusable after their compiled read");
    assert_eq!(int32_values(repeated.output().as_host_arrow().unwrap(), 1), vec![Some(200), Some(400)]);
    let admitted =
        admit_query_plan(&passthrough_plan, context.statistical_aggregate_execution_mode(), context.planning_policy())
            .unwrap();
    let outcome = engine_execute_query_plan(
        &context,
        QueryExecutionRequest::admitted(&admitted).with_owned_gpu_inputs(passthrough_inputs),
    );
    let (owned, _, _) = outcome.into_parts();
    let owned = owned.expect("owned inputs use the same compiled execution contract");
    assert_eq!(int32_values(owned.as_host_arrow().unwrap(), 1), vec![Some(200), Some(400)]);
}

fn compiled_gpu_executor_retires_owned_input_after_last_source_read_and_preserves_aliases() {
    struct InputRetirementExecutor<'a> {
        inner: CompiledGpuTaskExecutor<'a>,
        identity: &'a Arc<ProducerExecutionIdentity>,
        source_owner_counts: Vec<usize>,
    }

    impl CompiledTaskExecutor for InputRetirementExecutor<'_> {
        type Resource = CompiledGpuResource;

        fn execute_task(
            &mut self,
            task: &PipelineTask,
            state: &mut CompiledPipelineState<Self::Resource>,
        ) -> Result<Vec<(ResidentResourceId, Self::Resource)>> {
            let outputs = self.inner.execute_task(task, state)?;
            if task.kind() == PipelineTaskKind::GpuResidentSource {
                self.source_owner_counts.push(Arc::strong_count(self.identity));
            }
            Ok(outputs)
        }

        fn release_resource(&mut self, resource: ResidentResourceId, payload: Self::Resource) -> Result<()> {
            self.inner.release_resource(resource, payload)
        }

        fn cleanup_after_error(&mut self) -> Result<()> {
            self.inner.cleanup_after_error()
        }
    }

    let runtime = shared_runtime(suite_backend::Runtime::new().expect("compiled resident input requires CUDA"));
    let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int32, false)]));
    let mut builder = QueryPlanBuilder::new();
    let first = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", Arc::clone(&schema))));
    let second = builder.add_source(SourceNode::GpuResident(
        GpuResidentSource::new("input", Arc::clone(&schema)).with_row_count_upper_bound(3),
    ));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("value", DataType::Int32, false),
        Field::new("derived", DataType::Int32, false),
    ]));
    let mut branches = Vec::new();
    for (source, offset) in [(first, 1), (first, 2), (second, 3)] {
        let outputs = projection_outputs(
            vec![
                Expr::column("value"),
                Expr::binary(Expr::column("value"), BinaryOp::Add, Expr::literal(LiteralValue::Int32(offset))),
            ],
            Arc::clone(&output_schema),
        );
        branches.push(builder.add_projection(source, outputs).unwrap());
    }
    let union = builder.add_union(branches, output_schema).unwrap();
    let plan = builder.finish_with_host_sink(union).unwrap();
    let admitted = admit_query_plan(&plan, StatisticalAggregateExecutionMode::default(), PlanningPolicy::default())
        .expect("resident aliases and repeated source occurrences are supported");
    let context = bounded_context(Arc::clone(&runtime), &plan);
    let metrics = ExecutionMetrics::detached("owned_input_retirement");
    let metrics_context = metrics.scope_context(&context).unwrap();
    let input_batch =
        RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from(vec![3, 1, 2]))]).unwrap();
    let input = GpuResultHandle::new(
        &metrics_context,
        GpuDataFrame::new(
            CudfDataFrame::from_record_batch(metrics_context.shared_runtime().unwrap().as_ref(), &input_batch).unwrap(),
        ),
        schema,
    )
    .unwrap()
    .with_producer_execution_plan(plan.clone());
    let identity = input.producer_execution_identity().unwrap();
    let mut inputs = GpuResidentInputMap::new();
    inputs.insert("input".to_string(), input);
    let (_, inputs) = QueryExecutionRequest::admitted(&admitted).with_owned_gpu_inputs(inputs).into_parts();
    let executor = CompiledGpuTaskExecutor::new_owned(
        &metrics_context,
        &plan,
        admitted.policy(),
        admitted.pipeline(),
        inputs,
        &metrics,
    )
    .unwrap();
    let mut executor = InputRetirementExecutor { inner: executor, identity: &identity, source_owner_counts: vec![] };
    let scheduler = CompiledPipelineScheduler::new(admitted.pipeline()).unwrap();
    scheduler.run(&mut executor).expect("owned source aliases survive input retirement");
    let (last_owner_count, earlier_owner_counts) = executor.source_owner_counts.split_last().unwrap();
    assert!(!earlier_owner_counts.is_empty(), "distinct source bindings must exercise repeated input reads");
    assert!(earlier_owner_counts.iter().all(|count| *count == 2), "remaining readers retain the input owner");
    assert_eq!(*last_owner_count, 1, "the final scheduled read releases the input owner");
    let output = executor.inner.take_host_output().unwrap();
    let mut actual = int32_values(&output, 0).into_iter().zip(int32_values(&output, 1)).collect::<Vec<_>>();
    actual.sort();
    let expected =
        (1..=3).flat_map(|value| (1..=3).map(move |offset| (Some(value), Some(value + offset)))).collect::<Vec<_>>();
    assert_eq!(actual, expected);
    assert_eq!(metrics.snapshot().boundary.gpu_resident_input_handoffs, executor.source_owner_counts.len() as u64);

    let wrong_schema = Arc::new(Schema::new(vec![Field::new("other", DataType::Int32, false)]));
    let wrong_batch =
        RecordBatch::try_new(Arc::clone(&wrong_schema), vec![Arc::new(Int32Array::from(vec![3, 1, 2]))]).unwrap();
    let failed_input = GpuResultHandle::new(
        &metrics_context,
        GpuDataFrame::new(
            CudfDataFrame::from_record_batch(metrics_context.shared_runtime().unwrap().as_ref(), &wrong_batch).unwrap(),
        ),
        wrong_schema,
    )
    .unwrap()
    .with_producer_execution_plan(plan.clone());
    let failed_identity = failed_input.producer_execution_identity().unwrap();
    let mut failed_inputs = GpuResidentInputMap::new();
    failed_inputs.insert("input".to_string(), failed_input);
    let mut failed_executor = CompiledGpuTaskExecutor::new_owned(
        &metrics_context,
        &plan,
        admitted.policy(),
        admitted.pipeline(),
        failed_inputs,
        &metrics,
    )
    .unwrap();
    let error = scheduler.run(&mut failed_executor).err().expect("a mismatched resident source must fail");
    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.operation(), "query_engine.gpu_resident_source.read");
    assert_eq!(Arc::strong_count(&failed_identity), 1, "failed scheduler cleanup releases unread owned inputs");
}

#[cfg(feature = "cugraph")]
#[test]
fn graph_statistics_sizing_avoids_unused_work_and_counts_distinct_endpoint_union() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int64, false),
        Field::new("dst", DataType::Int64, false),
    ]));
    let graph_options = GraphConstructionOptions {
        directed: true,
        renumber: true,
        policy: GraphConstructionPolicy::RawLibcugraph,
        distinct_edge_pairs: false,
    };
    let params = GraphAlgorithmParams::PageRank(GraphPageRankParams::default());
    let pagerank = GraphAlgorithmSpec::new(
        GraphAlgorithm::PageRank,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        graph_options.clone(),
        params.clone(),
        graph_result_schema_for_vertex_type(GraphAlgorithm::PageRank, &params, &DataType::Int64),
    );
    let mut builder = QueryPlanBuilder::new();
    let edges = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("edges", Arc::clone(&schema))));
    let graph = builder.add_graph_algorithm(edges, vec![], pagerank.clone()).unwrap();
    let plan = builder.finish_with_host_sink(graph).unwrap();
    let runtime = shared_runtime(suite_backend::Runtime::new().expect("graph sizing requires CUDA"));
    let context = bounded_context(runtime, &plan);
    let mut sources = (0..65_536i64).map(|row| row % 9).collect::<Vec<_>>();
    let mut destinations = (0..65_536i64).map(|row| row % 12 + 5).collect::<Vec<_>>();
    sources.extend([i64::MIN, i64::MAX]);
    destinations.extend([i64::MAX, i64::MIN]);
    let expected_vertices = sources.iter().chain(&destinations).copied().collect::<BTreeSet<_>>().len() as u64;
    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(Int64Array::from(sources)), Arc::new(Int64Array::from(destinations))],
    )
    .unwrap();
    let frame = CudfDataFrame::from_record_batch(context.shared_runtime().unwrap().as_ref(), &batch).unwrap();
    let allocations_before = context.attempt_allocation_domain_usage().unwrap().unwrap().total_allocations;
    assert_eq!(runtime_graph_statistics_probe(&context, &frame, &pagerank).unwrap(), None);
    let allocations_after = context.attempt_allocation_domain_usage().unwrap().unwrap().total_allocations;
    assert_eq!(allocations_after, allocations_before, "PageRank has no runtime seed sizing work");

    let params = GraphAlgorithmParams::BetweennessCentrality(GraphBetweennessCentralityParams {
        seeds: Some(GraphBetweennessSeeds::I64(vec![0])),
        ..Default::default()
    });
    let mut betweenness = GraphAlgorithmSpec::new(
        GraphAlgorithm::BetweennessCentrality,
        GraphEdgeColumnBindings::new("src", "dst", None::<&str>),
        graph_options,
        params.clone(),
        graph_result_schema_for_vertex_type(GraphAlgorithm::BetweennessCentrality, &params, &DataType::Int64),
    );
    assert_eq!(runtime_graph_statistics_probe(&context, &frame, &betweenness).unwrap(), None);
    assert_eq!(context.attempt_allocation_domain_usage().unwrap().unwrap().total_allocations, allocations_before);
    betweenness.params = GraphAlgorithmParams::BetweennessCentrality(GraphBetweennessCentralityParams {
        exact_vertex_threshold: 1,
        ..Default::default()
    });
    let copies_before = device_to_device_copy_count_for_test();
    let count = runtime_graph_statistics_probe(&context, &frame, &betweenness).unwrap().unwrap();
    assert_eq!(count, expected_vertices);
    assert_eq!(device_to_device_copy_count_for_test(), copies_before, "endpoint sizing consumes the retained input");
    let error = validate_betweenness_seed_policy_for_vertex_count(&betweenness, count).unwrap_err();
    assert_eq!(error.fact_value("reason"), Some("exact_betweenness_vertex_threshold_exceeded"));
    assert_eq!(error.fact_value("actual_vertex_count"), Some(expected_vertices.to_string().as_str()));
}

/// Narrowing a uniquely owned edge frame holds at most one Int32 column above
/// the input frame, because each Int64 column is released once its Int32 copy
/// exists.
#[cfg(feature = "cugraph")]
#[test]
fn graph_vertex_id_narrowing_releases_each_int64_column_after_its_int32_copy() {
    const ROWS: usize = 1 << 20;
    const SLACK_BYTES: u64 = 1 << 20;
    let schema = Arc::new(Schema::new(vec![
        Field::new("src", DataType::Int64, false),
        Field::new("dst", DataType::Int64, false),
        Field::new("weight", DataType::Float64, false),
    ]));
    let mut builder = QueryPlanBuilder::new();
    let edges = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("edges", Arc::clone(&schema))));
    let plan = builder.finish_with_host_sink(edges).unwrap();
    let runtime = shared_runtime(suite_backend::Runtime::new().expect("graph id narrowing requires CUDA"));
    let context = bounded_context(runtime, &plan);
    let sources = (0..ROWS as i64).collect::<Vec<_>>();
    let destinations = (0..ROWS as i64).map(|row| (row * 7) % ROWS as i64).collect::<Vec<_>>();
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(sources.clone())),
            Arc::new(Int64Array::from(destinations.clone())),
            Arc::new(Float64Array::from(vec![1.0; ROWS])),
        ],
    )
    .unwrap();
    let frame = CudfDataFrame::from_record_batch(context.shared_runtime().unwrap().as_ref(), &batch).unwrap();
    let before = context.attempt_allocation_domain_usage().unwrap().unwrap();
    assert!(before.peak_bytes <= before.current_bytes + SLACK_BYTES, "{before:?}");

    let narrowing_runtime = context.runtime().unwrap();
    let narrowed = narrow_id_columns(GpuDataFrame::new(frame), &["src", "dst"], narrowing_runtime.exec()).unwrap();

    let after = context.attempt_allocation_domain_usage().unwrap().unwrap();
    let int32_column_bytes = (ROWS * size_of::<i32>()) as u64;
    assert!(after.peak_bytes <= before.current_bytes + int32_column_bytes + SLACK_BYTES, "{before:?} {after:?}");
    assert_eq!(after.current_bytes, before.current_bytes - 2 * int32_column_bytes, "{before:?} {after:?}");
    let column = |name| narrowed.frame().series(name).unwrap();
    assert_eq!(column("weight").dtype().unwrap().id(), cudf::TypeId::Float64);
    for (name, expected) in [("src", sources), ("dst", destinations)] {
        assert_eq!(column(name).dtype().unwrap().id(), cudf::TypeId::Int32, "{name}");
        assert!(column(name).to_vec_strict::<i32>().unwrap().into_iter().map(i64::from).eq(expected), "{name}");
    }
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
            PipelineTask::new(*task, TaskPhysical::SourceRead(SourceReadPlan::FRAME)).with_output_resources(outputs)
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

fn execute_query_plan_routes_root_hash_join_aggregate_through_compiled_pipeline_case() {
    let runtime = suite_backend::Runtime::new().expect("compiled GPU execution test requires a CUDA runtime");
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

fn execute_query_plan_routes_root_hash_join_aggregate_through_compiled_pipeline() {
    execute_query_plan_routes_root_hash_join_aggregate_through_compiled_pipeline_case();
    execute_query_plan_routes_single_join_post_root_graph_through_compiled_pipeline();
}

fn execute_query_plan_sizes_compiled_stream_aggregate_string_group_keys_inside_reservations() {
    let runtime = suite_backend::Runtime::new().expect("compiled GPU execution test requires a CUDA runtime");
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

fn execute_query_plan_routes_single_join_post_root_graph_through_compiled_pipeline() {
    let runtime = suite_backend::Runtime::new().expect("compiled GPU execution test requires a CUDA runtime");
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
    let runtime = suite_backend::Runtime::new().expect("compiled GPU execution test requires a CUDA runtime");
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

fn execute_query_plan_streams_residual_one_shot_probe_from_prior_join() {
    let runtime = suite_backend::Runtime::new().expect("compiled GPU execution test requires a CUDA runtime");
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
fn gpu_execution_cases_3() {
    run_named_cases(&[
        (
            "exec::bounded::compiled::execution_tests::gpu_execution::execute_query_plan_streams_residual_one_shot_probe_from_prior_join",
            execute_query_plan_streams_residual_one_shot_probe_from_prior_join,
        ),
        (
            "exec::bounded::compiled::execution_tests::gpu_execution::execute_query_plan_applies_limit_offset_once_through_fused_topk",
            execute_query_plan_applies_limit_offset_once_through_fused_topk,
        ),
        (
            "exec::bounded::compiled::execution_tests::gpu_execution::observed_compiled_root_join_records_registered_tasks_without_reservation_events",
            observed_compiled_root_join_records_registered_tasks_without_reservation_events,
        ),
        (
            "exec::bounded::compiled::execution_tests::gpu_execution::execute_query_plan_streams_residual_one_shot_right_probe_with_left_build",
            execute_query_plan_streams_residual_one_shot_right_probe_with_left_build,
        ),
        (
            "exec::bounded::compiled::execution_tests::gpu_execution::execute_query_plan_runs_a_union_that_reads_one_producer_twice",
            execute_query_plan_runs_a_union_that_reads_one_producer_twice,
        ),
        (
            "exec::bounded::compiled::execution_tests::gpu_execution::execute_query_plan_runs_independent_union_branches_through_unified_admission",
            execute_query_plan_runs_independent_union_branches_through_unified_admission,
        ),
        (
            "exec::bounded::compiled::execution_tests::gpu_execution::compiled_gpu_executor_retires_owned_input_after_last_source_read_and_preserves_aliases",
            compiled_gpu_executor_retires_owned_input_after_last_source_read_and_preserves_aliases,
        ),
    ]);
}

fn execute_query_plan_streams_residual_one_shot_probe_from_leaf_local_source() {
    let runtime = suite_backend::Runtime::new().expect("compiled GPU execution test requires a CUDA runtime");
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

fn execute_query_plan_streams_residual_one_shot_right_probe_with_left_build() {
    let runtime = suite_backend::Runtime::new().expect("compiled GPU execution test requires a CUDA runtime");
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
    let runtime = suite_backend::Runtime::new().expect("compiled GPU execution test requires a CUDA runtime");
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

/// One edge row: source, destination, weight and edge id.
type EdgeRow = (i64, i64, Option<f64>, Option<i64>);

struct EdgeNormalizeCase {
    directed: bool,
    distinct_edge_pairs: bool,
    weighted: bool,
    edge_ids: bool,
    vertex_type: DataType,
    rows: Vec<EdgeRow>,
}

impl EdgeNormalizeCase {
    /// An unweighted case ignores the weights in `rows`.
    fn new(directed: bool, weighted: bool, mut rows: Vec<EdgeRow>) -> Self {
        if !weighted {
            rows.iter_mut().for_each(|row| row.2 = None);
        }
        Self { directed, distinct_edge_pairs: false, weighted, edge_ids: true, vertex_type: DataType::Int64, rows }
    }

    /// Drops the edge-id column, and with it the edge ids in `rows`.
    fn without_edge_ids(mut self) -> Self {
        self.edge_ids = false;
        self.rows.iter_mut().for_each(|row| row.3 = None);
        self
    }

    fn spec(&self) -> EdgeNormalizeSpec {
        let mut edge_columns = GraphEdgeColumnBindings::new("src", "dst", self.weighted.then_some("weight"));
        if self.edge_ids {
            edge_columns = edge_columns.with_edge_id("edge_id");
        }
        EdgeNormalizeSpec::python_cugraph(
            edge_columns,
            &GraphConstructionOptions {
                directed: self.directed,
                distinct_edge_pairs: self.distinct_edge_pairs,
                ..Default::default()
            },
        )
    }

    /// The Python-compatible result: reversed rows appended for an undirected
    /// graph, then for each pair the first or last row of a stable sort by
    /// source, destination and, for a minimum-weight combiner, weight with
    /// nulls last.
    fn expected(&self) -> Vec<EdgeRow> {
        let spec = self.spec();
        let mut rows = self.rows.clone();
        if spec.add_reverse {
            rows.extend(
                self.rows.iter().map(|&(source, destination, weight, edge_id)| (destination, source, weight, edge_id)),
            );
        }
        rows.sort_by(|left, right| {
            (left.0, left.1).cmp(&(right.0, right.1)).then_with(|| {
                if spec.weight_combiner != EdgeWeightCombiner::MinWeight {
                    return std::cmp::Ordering::Equal;
                }
                match (left.2, right.2) {
                    (Some(left), Some(right)) => left.partial_cmp(&right).expect("test weights are not NaN"),
                    (left, right) => left.is_none().cmp(&right.is_none()),
                }
            })
        });
        let kept = rows
            .chunk_by(|left, right| (left.0, left.1) == (right.0, right.1))
            .map(|group| match spec.dedup {
                EdgeDedupStrategy::KeepFirst => group[0],
                EdgeDedupStrategy::KeepLast => group[group.len() - 1],
                EdgeDedupStrategy::None => unreachable!("python_cugraph always deduplicates"),
            })
            .collect();
        sorted_edge_rows(kept)
    }

    fn execute(&self, runtime: &Arc<suite_backend::Runtime>) -> Vec<EdgeRow> {
        let mut fields = vec![
            Field::new("src", self.vertex_type.clone(), false),
            Field::new("dst", self.vertex_type.clone(), false),
        ];
        let vertices = |values: Vec<i64>| -> ArrayRef {
            match self.vertex_type {
                DataType::Int32 => Arc::new(Int32Array::from(
                    values.into_iter().map(|value| i32::try_from(value).unwrap()).collect::<Vec<_>>(),
                )),
                _ => Arc::new(Int64Array::from(values)),
            }
        };
        let mut columns = vec![
            vertices(self.rows.iter().map(|row| row.0).collect()),
            vertices(self.rows.iter().map(|row| row.1).collect()),
        ];
        if self.weighted {
            fields.push(Field::new("weight", DataType::Float64, true));
            columns.push(Arc::new(Float64Array::from(self.rows.iter().map(|row| row.2).collect::<Vec<_>>())));
        }
        if self.edge_ids {
            fields.push(Field::new("edge_id", DataType::Int64, false));
            columns.push(Arc::new(Int64Array::from(self.rows.iter().map(|row| row.3.unwrap()).collect::<Vec<_>>())));
        }
        let schema = Arc::new(Schema::new(fields));
        let batch = RecordBatch::try_new(Arc::clone(&schema), columns).expect("edge literal batch");

        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(schema, vec![batch])));
        let normalize = builder.add_edge_normalize(source, self.spec()).unwrap();
        let plan = builder.finish_with_host_sink(normalize).unwrap();
        let context = runtime.acquire_context_for_plan(&plan).expect("edge plan acquires a context");
        let result = execute_query_plan(&context, &plan).expect("edge normalization should execute");
        let QueryOutput::HostArrow(output) = result.output() else {
            panic!("host sink should materialize one RecordBatch");
        };
        let vertex_values = |column: usize| -> Vec<i64> {
            match self.vertex_type {
                DataType::Int32 => {
                    int32_values(output, column).into_iter().map(|value| i64::from(value.unwrap())).collect()
                },
                _ => int64_values(output, column).into_iter().map(Option::unwrap).collect(),
            }
        };
        let weights = if self.weighted {
            let array = output.column(2).as_any().downcast_ref::<Float64Array>().unwrap();
            (0..array.len()).map(|row| array.is_valid(row).then(|| array.value(row))).collect()
        } else {
            vec![None; output.num_rows()]
        };
        let edge_ids =
            if self.edge_ids { int64_values(output, output.num_columns() - 1) } else { vec![None; output.num_rows()] };
        sorted_edge_rows(
            vertex_values(0)
                .into_iter()
                .zip(vertex_values(1))
                .zip(weights)
                .zip(edge_ids)
                .map(|(((source, destination), weight), edge_id)| (source, destination, weight, edge_id))
                .collect(),
        )
    }

    fn assert_matches_python_cugraph(&self, runtime: &Arc<suite_backend::Runtime>, label: &str) {
        assert_eq!(self.execute(runtime), self.expected(), "{label}");
    }
}

fn edge_normalize_runtime() -> Arc<suite_backend::Runtime> {
    shared_runtime(suite_backend::Runtime::new().expect("edge normalization GPU test requires a CUDA runtime"))
}

fn sorted_edge_rows(mut rows: Vec<EdgeRow>) -> Vec<EdgeRow> {
    rows.sort_by_key(|row| (row.0, row.1, row.3));
    rows
}

/// Both columns of the first input span the whole Int64 range, which needs
/// 128 bits directed and undirected, and sort on the endpoint columns. In the
/// second input the source spans `2^32 + 1` values and the destination one
/// value, and the directed rows pack into a 33-bit source field: `(0, 0)` and
/// `(2^32, 0)` stay apart, which a key that drops source bits above 32 would
/// merge.
fn edge_normalize_deduplicates_full_range_ids_and_a_33_bit_source() {
    let runtime = edge_normalize_runtime();
    let rows = vec![
        (i64::MIN, i64::MAX, None, Some(0)),
        (i64::MAX, i64::MIN, None, Some(1)),
        (i64::MIN, i64::MAX, None, Some(2)),
        (0, 5, None, Some(3)),
        (1_i64 << 32, 5, None, Some(4)),
        (-1, -1, None, Some(5)),
        (-(1_i64 << 40), 1_i64 << 40, None, Some(6)),
    ];
    EdgeNormalizeCase::new(false, false, rows.clone()).assert_matches_python_cugraph(&runtime, "undirected Int64");
    EdgeNormalizeCase::new(true, false, rows).assert_matches_python_cugraph(&runtime, "directed Int64");
    let two_pow_32 = vec![(0, 0, None, Some(0)), (1_i64 << 32, 0, None, Some(1)), (0, 0, None, Some(2))];
    EdgeNormalizeCase::new(true, false, two_pow_32).assert_matches_python_cugraph(&runtime, "33-bit source");
}

/// Int32 ids from `i32::MIN` to `i32::MAX` span `2^32 - 1`, the widest range
/// that still packs. Their offsets overflow an Int32 subtraction.
fn edge_normalize_packs_int32_ids_spanning_the_whole_type() {
    let runtime = edge_normalize_runtime();
    let rows = vec![
        (i64::from(i32::MIN), i64::from(i32::MAX), None, Some(0)),
        (i64::from(i32::MAX), i64::from(i32::MIN), None, Some(1)),
        (i64::from(i32::MAX), i64::from(i32::MAX), None, Some(2)),
        (-1, 0, None, Some(3)),
        (0, -1, None, Some(4)),
        (i64::from(i32::MIN), i64::from(i32::MAX), None, Some(5)),
    ];
    let mut case = EdgeNormalizeCase::new(false, false, rows);
    case.vertex_type = DataType::Int32;
    case.assert_matches_python_cugraph(&runtime, "undirected Int32");
}

/// An undirected graph keeps the minimum weight of each pair after reverse
/// expansion. Nulls sort after every weight, a pair with only null weights
/// keeps a null, and equal minimums keep the earlier row.
fn edge_normalize_keeps_the_first_minimum_weight_with_nulls_last() {
    let runtime = edge_normalize_runtime();
    let rows = vec![
        (1, 2, Some(5.0), Some(0)),
        (2, 1, Some(3.0), Some(1)),
        (1, 2, None, Some(2)),
        (3, 4, None, Some(3)),
        (4, 3, None, Some(4)),
        (5, 6, Some(2.0), Some(5)),
        (6, 5, Some(2.0), Some(6)),
        (5, 6, Some(2.0), Some(7)),
        (7, 7, Some(1.0), Some(8)),
        (8, 9, None, Some(9)),
        (9, 8, Some(4.0), Some(10)),
    ];
    EdgeNormalizeCase::new(false, true, rows).assert_matches_python_cugraph(&runtime, "undirected weighted");
}

/// A directed weighted graph keeps the last duplicate row in input order.
fn edge_normalize_keeps_the_last_directed_weighted_row() {
    let runtime = edge_normalize_runtime();
    let rows = vec![
        (1, 2, Some(5.0), Some(0)),
        (1, 2, Some(1.0), Some(1)),
        (2, 1, Some(7.0), Some(2)),
        (1, 2, None, Some(3)),
        (3, 3, Some(2.0), Some(4)),
        (3, 3, Some(9.0), Some(5)),
    ];
    EdgeNormalizeCase::new(true, true, rows).assert_matches_python_cugraph(&runtime, "directed weighted");
}

/// Distinct input pairs still need deduplication after reverse expansion when
/// the source and destination ranges overlap: `(1, 2)` meets the reverse of
/// `(2, 1)`. In the second input the ranges `[0, 5]` and `[5, 9]` share only
/// the value 5, and the self loop `(5, 5)` meets its own reverse.
fn edge_normalize_deduplicates_distinct_pairs_whose_ranges_overlap() {
    let runtime = edge_normalize_runtime();
    let crossing = vec![(1, 2, Some(4.0), Some(0)), (2, 1, Some(3.0), Some(1)), (0, 7, Some(1.0), Some(2))];
    let touching = vec![(0, 5, Some(2.0), Some(0)), (5, 5, None, Some(1)), (3, 9, Some(1.0), Some(2))];
    for (label, rows) in [("crossing", crossing), ("touching", touching)] {
        for weighted in [false, true] {
            let mut case = EdgeNormalizeCase::new(false, weighted, rows.clone());
            case.distinct_edge_pairs = true;
            case.assert_matches_python_cugraph(&runtime, &format!("{label} weighted={weighted}"));
        }
    }
}

/// Distinct input pairs whose source and destination ranges are disjoint
/// normalize to the input plus its reversed rows.
fn edge_normalize_appends_reverse_edges_to_distinct_pairs_with_disjoint_ranges() {
    let runtime = edge_normalize_runtime();
    let rows = vec![
        (100, 1, Some(2.0), Some(0)),
        (100, 2, None, Some(1)),
        (101, 1, Some(1.0), Some(2)),
        (102, 0, Some(3.0), Some(3)),
    ];
    for (directed, weighted) in [(false, false), (false, true), (true, true)] {
        let mut case = EdgeNormalizeCase::new(directed, weighted, rows.clone());
        case.distinct_edge_pairs = true;
        case.assert_matches_python_cugraph(&runtime, &format!("directed={directed} weighted={weighted}"));
    }
}

/// Without weight or edge-id columns the packed key is the whole frame. The
/// duplicates include a repeated pair, a pair that meets a reversed edge and a
/// self loop, for both vertex types.
fn edge_normalize_deduplicates_endpoint_only_edges() {
    let runtime = edge_normalize_runtime();
    let rows = vec![
        (3, 1, None, None),
        (1, 3, None, None),
        (3, 1, None, None),
        (2, 2, None, None),
        (-4, 7, None, None),
        (7, -4, None, None),
        (0, 1, None, None),
    ];
    for (vertex_type, directed) in
        [(DataType::Int64, false), (DataType::Int64, true), (DataType::Int32, false), (DataType::Int32, true)]
    {
        let mut case = EdgeNormalizeCase::new(directed, false, rows.clone()).without_edge_ids();
        case.vertex_type = vertex_type.clone();
        case.assert_matches_python_cugraph(&runtime, &format!("{vertex_type} directed={directed}"));
    }
}

/// Each column spans few bits, but the columns lie far apart. In the Int64
/// input the sources, offset by `10^10`, span 26 bits and the destinations 28
/// bits; the Int32 columns span 10 and 12 bits at opposite ends of the type.
/// `(u, 2^27)` and `(u + 2, 0)`, and their Int32 counterparts, stay apart,
/// which a key shifting the source by the source width would merge.
fn edge_normalize_packs_far_apart_ranges_per_column() {
    let runtime = edge_normalize_runtime();
    let user = 10_000_000_000_i64;
    let (last_user, last_item) = (user + (1 << 26) - 1, (1_i64 << 28) - 1);
    let int64_rows = vec![
        (user, 0, Some(2.0), Some(0)),
        (last_user, last_item, Some(1.0), Some(1)),
        (user, 1 << 27, None, Some(2)),
        (user + 2, 0, Some(4.0), Some(3)),
        (user, 0, Some(1.0), Some(4)),
        (last_user, 0, None, Some(5)),
        (user, last_item, Some(3.0), Some(6)),
        (last_user, last_item, Some(1.0), Some(7)),
    ];
    let (low, high) = (i64::from(i32::MIN), i64::from(i32::MAX));
    let int32_rows = vec![
        (low, high, Some(1.0), Some(0)),
        (low + 1023, high - 4095, None, Some(1)),
        (low, high, Some(1.0), Some(2)),
        (low, high - 3071, Some(5.0), Some(3)),
        (low + 1, high - 4095, Some(2.0), Some(4)),
        (low + 1023, high - 4095, Some(2.0), Some(5)),
    ];
    for (vertex_type, rows) in [(DataType::Int64, int64_rows), (DataType::Int32, int32_rows)] {
        for (directed, weighted) in [(true, false), (true, true), (false, false), (false, true)] {
            let mut case = EdgeNormalizeCase::new(directed, weighted, rows.clone());
            case.vertex_type = vertex_type.clone();
            let label = format!("{vertex_type} directed={directed} weighted={weighted}");
            case.assert_matches_python_cugraph(&runtime, &label);
        }
    }
}

/// A 40-bit source beside a 24-bit destination fills a 64-bit key, and the
/// keys of `base + 2^39` set bit 63. Beside a 25-bit destination the rows
/// sort on the endpoint columns. A 65-bit key would push the top source bit
/// of `(base + 2^39, 0)` out and merge the row with `(base, 0)`.
fn edge_normalize_packs_field_widths_summing_to_64_bits() {
    let runtime = edge_normalize_runtime();
    let base = 1_i64 << 50;
    for destination_bits in [24, 25] {
        let last = (1_i64 << destination_bits) - 1;
        let rows = vec![
            (base, 0, None, Some(0)),
            (base + (1 << 39), 0, None, Some(1)),
            (base + (1 << 40) - 1, last, None, Some(2)),
            (base, last, None, Some(3)),
            (base + (1 << 39), 0, None, Some(4)),
            (base, 1 << (destination_bits - 1), None, Some(5)),
        ];
        for directed in [true, false] {
            let label = format!("destination_bits={destination_bits} directed={directed}");
            EdgeNormalizeCase::new(directed, false, rows.clone()).assert_matches_python_cugraph(&runtime, &label);
        }
    }
}

/// A single-valued column takes 0 bits. Beside it a 63-bit destination packs
/// with a shift of 63, and a 64-bit source packs with a shift of 0 and offsets
/// up to `2^64 - 1`. A 64-bit destination would shift by 64 and sorts on the
/// endpoint columns.
fn edge_normalize_packs_single_valued_columns() {
    let runtime = edge_normalize_runtime();
    let negative_destinations = vec![
        (7, i64::MIN, None, Some(0)),
        (7, -1, None, Some(1)),
        (7, i64::MIN, None, Some(2)),
        (7, i64::MIN + (1 << 62), None, Some(3)),
    ];
    let full_range_sources = vec![
        (i64::MIN, 3, None, Some(0)),
        (i64::MAX, 3, None, Some(1)),
        (0, 3, None, Some(2)),
        (i64::MAX, 3, None, Some(3)),
        (-1, 3, None, Some(4)),
    ];
    let full_range_destinations = vec![
        (7, i64::MIN, None, Some(0)),
        (7, i64::MAX, None, Some(1)),
        (7, i64::MIN, None, Some(2)),
        (7, 0, None, Some(3)),
    ];
    let cases = [
        ("63-bit destination", negative_destinations.clone(), true),
        ("63-bit destination", negative_destinations, false),
        ("64-bit source", full_range_sources, true),
        ("64-bit destination", full_range_destinations, true),
    ];
    for (label, rows, directed) in cases {
        let label = format!("{label} directed={directed}");
        EdgeNormalizeCase::new(directed, false, rows).assert_matches_python_cugraph(&runtime, &label);
    }
}

/// With disjoint ranges and no distinct-pair proof, the input rows are
/// deduplicated before the reversed rows are appended. Each pair keeps the
/// first minimum weight with nulls last, or the last row when unweighted, and
/// its reversed pair keeps the reversed copy of that row. The narrow ranges
/// pack. The wide ones, at `i64::MIN` and `i64::MAX`, need 73 bits and sort on
/// the endpoint columns.
fn edge_normalize_deduplicates_input_rows_before_appending_reversed_rows() {
    let runtime = edge_normalize_runtime();
    let rows = |source: i64, destination: i64, extra: (i64, i64)| {
        vec![
            (source, destination, Some(5.0), Some(0)),
            (source, destination, Some(3.0), Some(1)),
            (source, destination, None, Some(2)),
            (source, destination, Some(3.0), Some(3)),
            (source + 1, destination, None, Some(4)),
            (source + 1, destination, None, Some(5)),
            (source + 1, destination + 1, Some(2.0), Some(6)),
            (source, destination + 1, None, Some(7)),
            (source, destination + 1, Some(4.0), Some(8)),
            (extra.0, extra.1, Some(1.0), Some(9)),
        ]
    };
    let narrow = rows(100, 10, (102, 8));
    let wide = rows(i64::MIN, i64::MAX - 1, (i64::MIN + (1 << 40), i64::MAX - (1 << 31)));
    for (label, rows) in [("narrow", narrow), ("wide", wide)] {
        for weighted in [false, true] {
            let label = format!("{label} weighted={weighted}");
            EdgeNormalizeCase::new(false, weighted, rows.clone()).assert_matches_python_cugraph(&runtime, &label);
        }
    }
}

/// Ranges that share one value overlap, and the rows are deduplicated with
/// their reversed rows included: the repeated self loop `(5, 5)` meets its
/// own reverse and keeps one row.
fn edge_normalize_deduplicates_ranges_that_share_one_value() {
    let runtime = edge_normalize_runtime();
    let rows = vec![
        (0, 5, Some(2.0), Some(0)),
        (5, 5, None, Some(1)),
        (5, 5, Some(1.0), Some(2)),
        (3, 9, Some(1.0), Some(3)),
        (0, 5, Some(1.0), Some(4)),
    ];
    for weighted in [false, true] {
        let label = format!("weighted={weighted}");
        EdgeNormalizeCase::new(false, weighted, rows.clone()).assert_matches_python_cugraph(&runtime, &label);
    }
}

fn edge_normalize_returns_no_rows_for_an_empty_input() {
    let runtime = edge_normalize_runtime();
    for distinct_edge_pairs in [false, true] {
        let mut case = EdgeNormalizeCase::new(false, true, Vec::new());
        case.distinct_edge_pairs = distinct_edge_pairs;
        case.assert_matches_python_cugraph(&runtime, &format!("distinct_edge_pairs={distinct_edge_pairs}"));
    }
}

#[test]
fn edge_normalize_cases() {
    run_named_cases(&[
        (
            "exec::bounded::compiled::execution_tests::gpu_execution::edge_normalize_deduplicates_full_range_ids_and_a_33_bit_source",
            edge_normalize_deduplicates_full_range_ids_and_a_33_bit_source,
        ),
        (
            "exec::bounded::compiled::execution_tests::gpu_execution::edge_normalize_packs_int32_ids_spanning_the_whole_type",
            edge_normalize_packs_int32_ids_spanning_the_whole_type,
        ),
        (
            "exec::bounded::compiled::execution_tests::gpu_execution::edge_normalize_keeps_the_first_minimum_weight_with_nulls_last",
            edge_normalize_keeps_the_first_minimum_weight_with_nulls_last,
        ),
        (
            "exec::bounded::compiled::execution_tests::gpu_execution::edge_normalize_keeps_the_last_directed_weighted_row",
            edge_normalize_keeps_the_last_directed_weighted_row,
        ),
        (
            "exec::bounded::compiled::execution_tests::gpu_execution::edge_normalize_deduplicates_distinct_pairs_whose_ranges_overlap",
            edge_normalize_deduplicates_distinct_pairs_whose_ranges_overlap,
        ),
        (
            "exec::bounded::compiled::execution_tests::gpu_execution::edge_normalize_appends_reverse_edges_to_distinct_pairs_with_disjoint_ranges",
            edge_normalize_appends_reverse_edges_to_distinct_pairs_with_disjoint_ranges,
        ),
        (
            "exec::bounded::compiled::execution_tests::gpu_execution::edge_normalize_deduplicates_endpoint_only_edges",
            edge_normalize_deduplicates_endpoint_only_edges,
        ),
        (
            "exec::bounded::compiled::execution_tests::gpu_execution::edge_normalize_packs_far_apart_ranges_per_column",
            edge_normalize_packs_far_apart_ranges_per_column,
        ),
        (
            "exec::bounded::compiled::execution_tests::gpu_execution::edge_normalize_packs_field_widths_summing_to_64_bits",
            edge_normalize_packs_field_widths_summing_to_64_bits,
        ),
        (
            "exec::bounded::compiled::execution_tests::gpu_execution::edge_normalize_packs_single_valued_columns",
            edge_normalize_packs_single_valued_columns,
        ),
        (
            "exec::bounded::compiled::execution_tests::gpu_execution::edge_normalize_deduplicates_input_rows_before_appending_reversed_rows",
            edge_normalize_deduplicates_input_rows_before_appending_reversed_rows,
        ),
        (
            "exec::bounded::compiled::execution_tests::gpu_execution::edge_normalize_deduplicates_ranges_that_share_one_value",
            edge_normalize_deduplicates_ranges_that_share_one_value,
        ),
        (
            "exec::bounded::compiled::execution_tests::gpu_execution::edge_normalize_returns_no_rows_for_an_empty_input",
            edge_normalize_returns_no_rows_for_an_empty_input,
        ),
    ]);
}
