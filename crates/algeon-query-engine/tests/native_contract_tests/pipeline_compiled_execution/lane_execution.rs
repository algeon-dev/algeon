use query_engine::exec::{LaneDispatchSupport, is_lane_family_pipeline, run_lane_dispatched};

use super::*;

// `cudf::Runtime` explicitly implements Send and Sync: it re-enters its
// device at each FFI boundary. Clippy cannot see that cross-crate unsafe impl.
#[expect(clippy::arc_with_non_send_sync, reason = "cudf::Runtime explicitly implements Send and Sync")]
fn lane_shared_runtime(runtime: cudf::Runtime) -> Arc<cudf::Runtime> {
    Arc::new(runtime)
}

/// Width 1, width 2, and the serial scheduler agree on independent union
/// branches. Width 1 runs the lane protocol inline, width 2 runs it on pool
/// blocking workers; the serial run is the pre-lane baseline. All three share
/// `execute_task`, the claim/output-ready/commit protocol, and the release
/// paths, so any divergence is a lane-dispatch defect.
#[test]
fn lane_dispatch_width_1_and_2_agree_with_serial_on_union_branches() {
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

    let decision = bounded_pipeline_decision(
        &plan,
        StatisticalAggregateExecutionMode::default(),
        query_engine::rapids_backend::rapids_analysis_context(),
    )
    .expect("bounded admission should not error");
    let admission = decision.admission().expect("independent union branches use the same compiled admission");
    let pipeline = admission.pipeline().clone();
    assert!(is_lane_family_pipeline(&pipeline), "literal union branches dispatch on lanes");

    let runtime = lane_shared_runtime(runtime);
    let test_context = runtime.acquire_context_for_plan(&plan).expect("admitted union plan acquires a context");
    let route = test_context.attempt_stream_provider_route().expect("admitted context carries a fork-stream route");
    assert!(route.lane_capacity() > 0, "an admitted attempt owns at least one fork stream");

    let run = |requested_lanes: Option<usize>| {
        let metrics = query_engine::exec::ExecutionMetrics::detached("lane-parity");
        let metrics_context = metrics.scope_context(&test_context).expect("metrics scope applies");
        let gpu_inputs = GpuResidentInputMap::new();
        let scheduler = CompiledPipelineScheduler::new(&pipeline).expect("pipeline should be schedulable");
        let mut executor =
            CompiledGpuTaskExecutor::new(&metrics_context, &plan, &pipeline, &gpu_inputs, &metrics).unwrap();
        match requested_lanes {
            None => {
                let state = scheduler.run(&mut executor).expect("serial union branches should execute");
                assert_eq!(state.live_resource_count(), 0, "serial run releases every resource");
            },
            Some(width) => {
                let support = LaneDispatchSupport::new(
                    route.clone(),
                    metrics_context.clone(),
                    plan.clone(),
                    pipeline.clone(),
                    gpu_inputs.clone(),
                    metrics.clone(),
                )
                .with_requested_lanes(width);
                let state = run_lane_dispatched(&scheduler, &mut executor, None, &support)
                    .expect("lane-dispatched union branches should execute");
                assert_eq!(state.live_resource_count(), 0, "lane run releases every resource");
            },
        }
        let batch = executor.take_host_output().expect("compiled host sink should materialize");
        let mut values = int32_values(&batch, 0);
        values.sort();
        values
    };

    let expected = vec![Some(1), Some(2), Some(3)];
    assert_eq!(run(None), expected, "serial baseline computes every branch input once");
    assert_eq!(run(Some(1)), expected, "width 1 runs the same lane protocol inline");
    assert_eq!(run(Some(2)), expected, "width 2 computes every branch input exactly once");
}

/// Two materialized parquet scan/projection branches under union and limit,
/// through the production query path. This is the overlap workload: both
/// branches carry real decode, projection, and concat kernels on disjoint
/// fork subsets.
#[test]
fn lane_dispatch_executes_parquet_union_branches_on_production_path() {
    let runtime = cudf::Runtime::new().expect("compiled GPU execution test requires a CUDA runtime");
    let dir = unique_temp_dir("query-engine-lane-parquet-union");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    write_key_value_parquet(&left_path, &[1, 2], &[10, 20]);
    write_key_value_parquet(&right_path, &[3, 4], &[30, 40]);
    let schema =
        Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false), Field::new("v", DataType::Int64, true)]));

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(fixture_local_parquet_source(&left_path, &schema)));
    let left = builder.add_projection(
        left,
        vec![
            ProjectionOutput::new(Expr::column("k"), Field::new("k", DataType::Int32, false)),
            ProjectionOutput::new(Expr::column("v"), Field::new("v", DataType::Int64, true)),
        ],
    );
    let right = builder.add_source(SourceNode::LocalParquet(fixture_local_parquet_source(&right_path, &schema)));
    let right = builder.add_projection(
        right,
        vec![
            ProjectionOutput::new(Expr::column("k"), Field::new("k", DataType::Int32, false)),
            ProjectionOutput::new(Expr::column("v"), Field::new("v", DataType::Int64, true)),
        ],
    );
    let union = builder.add_union(vec![left, right], Arc::clone(&schema));
    let limit = builder.add_limit(union, LimitSpec::fetch(10), Arc::clone(&schema));
    let plan = builder.finish_with_host_sink(limit);

    let decision = bounded_pipeline_decision(
        &plan,
        StatisticalAggregateExecutionMode::default(),
        query_engine::rapids_backend::rapids_analysis_context(),
    )
    .expect("bounded admission should not error");
    let admission = decision.admission().expect("parquet union branches use the compiled admission");
    let pipeline = admission.pipeline().clone();
    assert!(
        pipeline.tasks().iter().any(|task| task.kind() == PipelineTaskKind::Union),
        "both branches compile into one pipeline behind a union task"
    );
    assert!(
        pipeline.tasks().iter().any(|task| task.kind() == PipelineTaskKind::Limit),
        "the trailing limit stays in the lane family"
    );
    assert!(is_lane_family_pipeline(&pipeline));

    let runtime = lane_shared_runtime(runtime);
    let test_context = runtime.acquire_context_for_plan(&plan).expect("admitted union plan acquires a context");
    let result = execute_query_plan(&test_context, &plan).expect("lane-dispatched union plan should execute");
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(
        int32_int64_pairs_sorted(batch, 0, 1),
        vec![(Some(1), Some(10)), (Some(2), Some(20)), (Some(3), Some(30)), (Some(4), Some(40))],
        "every parquet branch row survives dispatch exactly once"
    );
}

fn fixture_local_parquet_source(path: &Path, schema: &Arc<Schema>) -> NativeLocalParquetSource {
    let file = LocalParquetFile::new(path);
    NativeLocalParquetSource::from_files(
        vec![enrich_fixture_local_parquet_file(file)],
        Arc::clone(schema),
        vec!["k".into(), "v".into()],
    )
}
