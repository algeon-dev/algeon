use std::{sync::Condvar, time::Duration};

use query_engine::exec::{
    LaneDispatchSupport, is_lane_family_pipeline, is_resident_aggregate_lane_pipeline, run_lane_dispatched,
};

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
                    Arc::clone(&route),
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

#[test]
fn lane_dispatch_scan_filter_aggregate_widths_preserve_rows_and_overlap_partials() {
    let runtime = cudf::Runtime::new().expect("compiled GPU execution test requires a CUDA runtime");
    let dir = unique_temp_dir("query-engine-lane-parquet-aggregate");
    let input_schema =
        Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false), Field::new("v", DataType::Int64, false)]));
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("rows", DataType::Int64, false),
        Field::new("minimum", DataType::Int64, false),
        Field::new("maximum", DataType::Int64, false),
    ]));
    let path = dir.join("input.parquet");
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(80_000)).build();
    let mut writer = ArrowWriter::try_new(File::create(&path).unwrap(), Arc::clone(&input_schema), Some(props))
        .expect("aggregate lane parquet writer");
    let mut counts = [0_i64; 4096];
    let mut minima = [i64::MAX; 4096];
    let mut maxima = [i64::MIN; 4096];
    for file_index in 0..4_u64 {
        let mut keys = Vec::with_capacity(160_000);
        let mut values = Vec::with_capacity(160_000);
        for row in 0..160_000_u64 {
            let key = if row % 10 < 8 { 0 } else { 1 + ((row / 10) % 4095) as usize };
            let value = aggregate_lane_value(row.wrapping_add(file_index.wrapping_mul(160_000)));
            keys.push(key as i32);
            values.push(value);
            if value >= 0 {
                counts[key] += 1;
                minima[key] = minima[key].min(value);
                maxima[key] = maxima[key].max(value);
            }
        }
        let batch = RecordBatch::try_new(
            Arc::clone(&input_schema),
            vec![Arc::new(Int32Array::from(keys)) as ArrayRef, Arc::new(Int64Array::from(values)) as ArrayRef],
        )
        .expect("aggregate lane input batch");
        writer.write(&batch).expect("aggregate lane parquet write");
    }
    writer.close().expect("aggregate lane parquet close");
    let source = NativeLocalParquetSource::from_files(
        vec![enrich_fixture_local_parquet_file(LocalParquetFile::new(path))],
        Arc::clone(&input_schema),
        vec!["k".to_string(), "v".to_string()],
    );
    let mut builder = QueryPlanBuilder::new();
    let scan = builder.add_source(SourceNode::LocalParquet(source.clone()));
    let filter = builder.add_filter(
        scan,
        Expr::binary(Expr::column("v"), BinaryOp::GtEq, Expr::literal(LiteralValue::Int64(0))),
        Arc::clone(&input_schema),
    );
    let projection = builder.add_projection(
        filter,
        vec![
            ProjectionOutput::new(Expr::column("k"), Field::new("k", DataType::Int32, false)),
            ProjectionOutput::new(Expr::column("v"), Field::new("measure", DataType::Int64, false)),
        ],
    );
    let aggregate = builder.add_aggregate(
        projection,
        AggregateSpec::new(
            Vec::new(),
            vec![
                AggregateExpr::new(AggregateFunction::Count, Expr::column("measure"), "rows"),
                AggregateExpr::new(AggregateFunction::Min, Expr::column("measure"), "minimum"),
                AggregateExpr::new(AggregateFunction::Max, Expr::column("measure"), "maximum"),
            ],
        ),
        Arc::clone(&aggregate_schema),
    );
    let plan = builder.finish_with_host_sink(aggregate);
    let decision = bounded_pipeline_decision(
        &plan,
        StatisticalAggregateExecutionMode::default(),
        query_engine::rapids_backend::rapids_analysis_context(),
    )
    .expect("scan/filter/aggregate admission should not error");
    let pipeline = decision.admission().expect("scan/filter/aggregate uses compiled admission").pipeline().clone();
    assert!(is_lane_family_pipeline(&pipeline));
    assert_eq!(pipeline.tasks().iter().filter(|task| task.kind() == PipelineTaskKind::Aggregate).count(), 1);

    let runtime = lane_shared_runtime(runtime);
    let test_context = runtime.acquire_context_for_plan(&plan).expect("aggregate plan acquires an admitted context");
    let route = test_context.attempt_stream_provider_route().expect("admitted aggregate owns a fork route");
    assert!(route.lane_capacity() >= 4, "the test backend must provision four source-work forks");
    let expected = vec![(
        Some(counts.iter().sum::<i64>()),
        Some(*minima.iter().min().unwrap()),
        Some(*maxima.iter().max().unwrap()),
    )];
    let arrivals = Arc::new((Mutex::new(Vec::new()), Condvar::new()));
    let run = |width: usize, coordinate: bool| {
        let metrics = query_engine::exec::ExecutionMetrics::detached("aggregate-lane-parity");
        let metrics_context = metrics.scope_context(&test_context).expect("metrics scope applies");
        let gpu_inputs = GpuResidentInputMap::new();
        let scheduler = CompiledPipelineScheduler::new(&pipeline).expect("aggregate pipeline is schedulable");
        let mut executor =
            CompiledGpuTaskExecutor::new(&metrics_context, &plan, &pipeline, &gpu_inputs, &metrics).unwrap();
        let mut support = LaneDispatchSupport::new(
            Arc::clone(&route),
            metrics_context.clone(),
            plan.clone(),
            pipeline.clone(),
            gpu_inputs.clone(),
            metrics.clone(),
        )
        .with_requested_lanes(width);
        if coordinate {
            let arrivals = Arc::clone(&arrivals);
            support = support.with_aggregate_chunk_completion_gate(Arc::new(move |ordinal, lane, fork| {
                let (mutex, wake) = &*arrivals;
                let mut observed = mutex.lock().expect("aggregate arrival mutex");
                observed.push((ordinal, lane, fork));
                wake.notify_all();
                let (observed, _) = wake
                    .wait_timeout_while(observed, Duration::from_secs(10), |observed| observed.len() < 2)
                    .expect("aggregate arrival wait");
                assert!(observed.len() >= 2, "two partial workers must reach completion before either can return");
            }));
        }
        let state = run_lane_dispatched(&scheduler, &mut executor, None, &support)
            .expect("compiled aggregate lanes should complete");
        assert_eq!(state.live_resource_count(), 0, "aggregate run releases every resource");
        let batch = executor.take_host_output().expect("aggregate sink materializes one batch");
        let counts = int64_values(&batch, 0);
        let minima = int64_values(&batch, 1);
        let maxima = int64_values(&batch, 2);
        let mut rows = counts
            .into_iter()
            .zip(minima)
            .zip(maxima)
            .map(|((count, minimum), maximum)| (count, minimum, maximum))
            .collect::<Vec<_>>();
        rows.sort();
        rows
    };
    assert_eq!(run(1, false), expected, "width 1 counts every filtered source row once");
    assert_eq!(run(2, true), expected, "width 2 agrees with the same oracle");
    assert_eq!(run(4, false), expected, "width 4 agrees with the same oracle");
    let observed = arrivals.0.lock().expect("aggregate arrival mutex");
    assert!(observed.len() >= 2);
    assert_ne!(observed[0].0, observed[1].0, "partial work has distinct chunk ordinals");
    assert_ne!(observed[0].1, observed[1].1, "partial work occupies distinct lanes");
    assert_ne!(observed[0].2, observed[1].2, "partial work owns distinct fork streams");
    drop(route);
    drop(test_context);

    let resident_plan = plan.clone_with_gpu_resident_sink();
    let resident_pipeline = bounded_pipeline_decision(
        &resident_plan,
        StatisticalAggregateExecutionMode::default(),
        query_engine::rapids_backend::rapids_analysis_context(),
    )
    .expect("resident aggregate admission should not error")
    .admission()
    .expect("resident aggregate uses compiled admission")
    .pipeline()
    .clone();
    assert!(is_resident_aggregate_lane_pipeline(&resident_pipeline));
    assert!(!is_lane_family_pipeline(&resident_pipeline));
    let resident_context = runtime.acquire_context_for_plan(&resident_plan).expect("resident plan acquires context");
    let result =
        execute_query_plan(&resident_context, &resident_plan).expect("resident aggregate lanes should execute");
    let QueryOutput::GpuResident(handle) = result.output() else {
        panic!("resident aggregate retains its output");
    };
    let handle = handle.clone();
    assert!(result.metrics().source_chunks_committed >= 2);
    drop(result);
    let batch = materialize_gpu_result_to_host(handle, &mut QueryMetrics::default())
        .expect("resident output remains readable after its lane leases are released");
    assert_eq!(int64_values(&batch, 0), vec![expected[0].0]);
    assert_eq!(int64_values(&batch, 1), vec![expected[0].1]);
    assert_eq!(int64_values(&batch, 2), vec![expected[0].2]);
    drop(resident_context);

    let grouped_schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("rows", DataType::Int64, false),
        Field::new("minimum", DataType::Int64, false),
        Field::new("maximum", DataType::Int64, false),
    ]));
    let mut builder = QueryPlanBuilder::new();
    let scan = builder.add_source(SourceNode::LocalParquet(source));
    let filter = builder.add_filter(
        scan,
        Expr::binary(Expr::column("v"), BinaryOp::GtEq, Expr::literal(LiteralValue::Int64(0))),
        Arc::clone(&input_schema),
    );
    let projection = builder.add_projection(
        filter,
        vec![
            ProjectionOutput::new(Expr::column("k"), Field::new("k", DataType::Int32, false)),
            ProjectionOutput::new(Expr::column("v"), Field::new("measure", DataType::Int64, false)),
        ],
    );
    let grouped = builder.add_aggregate(
        projection,
        AggregateSpec::new(
            vec![Expr::column("k")],
            vec![
                AggregateExpr::new(AggregateFunction::Count, Expr::column("measure"), "rows"),
                AggregateExpr::new(AggregateFunction::Min, Expr::column("measure"), "minimum"),
                AggregateExpr::new(AggregateFunction::Max, Expr::column("measure"), "maximum"),
            ],
        ),
        grouped_schema,
    );
    let grouped_plan = builder.finish_gpu_resident(grouped);
    let grouped_pipeline = bounded_pipeline_decision(
        &grouped_plan,
        StatisticalAggregateExecutionMode::default(),
        query_engine::rapids_backend::rapids_analysis_context(),
    )
    .expect("grouped aggregate admission should not error")
    .admission()
    .expect("grouped aggregate uses compiled admission")
    .pipeline()
    .clone();
    assert!(is_resident_aggregate_lane_pipeline(&grouped_pipeline));
    let grouped_context = runtime.acquire_context_for_plan(&grouped_plan).expect("grouped plan acquires context");
    let grouped_result =
        execute_query_plan(&grouped_context, &grouped_plan).expect("grouped aggregate lanes should execute");
    assert_eq!(grouped_result.metrics().source_chunks_committed, 8);
    assert!(grouped_result.metrics().partial_aggregate_count >= 8);
    let QueryOutput::GpuResident(handle) = grouped_result.output() else {
        panic!("grouped aggregate retains its output");
    };
    let handle = handle.clone();
    drop(grouped_result);
    let batch = materialize_gpu_result_to_host(handle, &mut QueryMetrics::default())
        .expect("grouped result remains readable after its lane leases are released");
    let mut rows = int32_values(&batch, 0)
        .into_iter()
        .zip(int64_values(&batch, 1))
        .zip(int64_values(&batch, 2))
        .zip(int64_values(&batch, 3))
        .map(|(((key, count), minimum), maximum)| (key, count, minimum, maximum))
        .collect::<Vec<_>>();
    rows.sort();
    assert_eq!(
        rows,
        (0..counts.len())
            .filter(|key| counts[*key] > 0)
            .map(|key| (Some(key as i32), Some(counts[key]), Some(minima[key]), Some(maxima[key])))
            .collect::<Vec<_>>()
    );
    drop(grouped_context);

    let single_source = NativeLocalParquetSource::from_files(
        vec![enrich_fixture_local_parquet_file(
            LocalParquetFile::new(dir.join("input.parquet")).with_row_groups(vec![0]),
        )],
        Arc::clone(&input_schema),
        vec!["k".to_string(), "v".to_string()],
    );
    let single_schema = Arc::new(Schema::new(vec![Field::new("rows", DataType::Int64, false)]));
    let mut builder = QueryPlanBuilder::new();
    let scan = builder.add_source(SourceNode::LocalParquet(single_source));
    let aggregate = builder.add_aggregate(
        scan,
        AggregateSpec::new(Vec::new(), vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("v"), "rows")]),
        single_schema,
    );
    let single_plan = builder.finish_gpu_resident(aggregate);
    let single_context = runtime.acquire_context_for_plan(&single_plan).expect("single chunk acquires context");
    let single_result =
        execute_query_plan(&single_context, &single_plan).expect("single chunk aggregate should execute");
    assert_eq!(single_result.metrics().source_chunks_committed, 1);
    let QueryOutput::GpuResident(handle) = single_result.output() else {
        panic!("single chunk aggregate retains its output");
    };
    assert_eq!(
        handle.frame().placement().bundle_id,
        handle.runtime().placement().bundle_id,
        "a lane-produced resident frame must be adopted onto the root runtime"
    );
    let handle = handle.clone();
    drop(single_result);
    let batch = materialize_gpu_result_to_host(handle, &mut QueryMetrics::default())
        .expect("single chunk output remains readable after its lane lease is released");
    assert_eq!(int64_values(&batch, 0), vec![Some(80_000)]);
}

fn aggregate_lane_value(index: u64) -> i64 {
    let mut value = index.wrapping_add(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
    (value ^ (value >> 31)) as i64
}

#[test]
fn lane_aggregate_sum_overflow_matches_the_single_lane_contract() {
    let runtime = cudf::Runtime::new().expect("compiled GPU execution test requires a CUDA runtime");
    let dir = unique_temp_dir("query-engine-lane-sum-overflow");
    let path = dir.join("input.parquet");
    let input_schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)]));
    let output_schema = Arc::new(Schema::new(vec![Field::new("total", DataType::Int64, true)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&input_schema),
        vec![Arc::new(Int64Array::from(vec![i64::MAX, 1, 0, 0])) as ArrayRef],
    )
    .expect("overflow input batch");
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(1)).build();
    let mut writer = ArrowWriter::try_new(File::create(&path).unwrap(), Arc::clone(&input_schema), Some(props))
        .expect("overflow input writer");
    writer.write(&batch).expect("overflow row groups");
    writer.close().expect("overflow input close");
    let source = NativeLocalParquetSource::from_files(
        vec![enrich_fixture_local_parquet_file(LocalParquetFile::new(path))],
        Arc::clone(&input_schema),
        vec!["v".to_string()],
    );
    let mut builder = QueryPlanBuilder::new();
    let scan = builder.add_source(SourceNode::LocalParquet(source));
    let aggregate = builder.add_aggregate(
        scan,
        AggregateSpec::new(Vec::new(), vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("v"), "total")]),
        output_schema,
    );
    let plan = builder.finish_with_host_sink(aggregate);
    let decision = bounded_pipeline_decision(
        &plan,
        StatisticalAggregateExecutionMode::default(),
        query_engine::rapids_backend::rapids_analysis_context(),
    )
    .expect("overflow aggregate admission should not error");
    let pipeline = decision.admission().expect("overflow aggregate uses compiled admission").pipeline().clone();
    assert!(is_lane_family_pipeline(&pipeline));

    let runtime = lane_shared_runtime(runtime);
    let test_context = runtime.acquire_context_for_plan(&plan).expect("overflow aggregate acquires context");
    let route = test_context.attempt_stream_provider_route().expect("overflow aggregate has a fork route");
    assert!(route.lane_capacity() >= 4);
    let run = |width: usize| -> std::result::Result<Vec<Option<i64>>, String> {
        let metrics = query_engine::exec::ExecutionMetrics::detached("aggregate-overflow-lane-parity");
        let metrics_context = metrics.scope_context(&test_context).expect("metrics scope applies");
        let gpu_inputs = GpuResidentInputMap::new();
        let scheduler = CompiledPipelineScheduler::new(&pipeline).expect("overflow pipeline is schedulable");
        let mut executor =
            CompiledGpuTaskExecutor::new(&metrics_context, &plan, &pipeline, &gpu_inputs, &metrics).unwrap();
        let support = LaneDispatchSupport::new(
            Arc::clone(&route),
            metrics_context.clone(),
            plan.clone(),
            pipeline.clone(),
            gpu_inputs.clone(),
            metrics.clone(),
        )
        .with_requested_lanes(width);
        match run_lane_dispatched(&scheduler, &mut executor, None, &support) {
            Ok(_) => Ok(int64_values(&executor.take_host_output().expect("overflow sink output"), 0)),
            Err(error) => Err(error.code().to_string()),
        }
    };
    let expected = run(1);
    assert_eq!(expected, Ok(vec![Some(i64::MIN)]), "single-lane SUM retains the existing wrapping result");
    assert_eq!(run(2), expected, "two lanes preserve the single-lane overflow outcome");
    assert_eq!(run(4), expected, "four lanes preserve the single-lane overflow outcome");
}
