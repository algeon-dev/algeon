use std::{sync::Condvar, time::Duration};

use ::cudf::testing::device_copy;
use arrow::array::{BooleanArray, Int64Array, StringArray};
use query_engine::{
    dimension_pruning::local_dimension_pruning_resolver,
    error::QueryErrorCode,
    exec::{
        CompiledGpuTaskExecutor, CompiledPipelineScheduler, ExecutionMetrics, LaneDispatchSupport,
        is_lane_family_pipeline, run_lane_dispatched,
    },
    optimizer::{NativeOptimizerOptions, optimize_query_plan},
    pipeline::SortStrategy,
    policy::PlanningPolicy,
    rapids_backend::rapids_analysis_context,
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

    let decision = admit(
        &plan,
        StatisticalAggregateExecutionMode::default(),
        PlanningPolicy::default(),
        rapids_analysis_context(),
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
        let metrics = ExecutionMetrics::detached("lane-parity");
        let metrics_context = metrics.scope_context(&test_context).expect("metrics scope applies");
        let gpu_inputs = GpuResidentInputMap::new();
        let scheduler = CompiledPipelineScheduler::new(&pipeline).expect("pipeline should be schedulable");
        let mut executor =
            CompiledGpuTaskExecutor::new(&metrics_context, &plan, admission.policy(), &pipeline, &gpu_inputs, &metrics)
                .unwrap();
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
                    admission.policy(),
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

    let decision = admit(
        &plan,
        StatisticalAggregateExecutionMode::default(),
        PlanningPolicy::default(),
        rapids_analysis_context(),
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
    let decision = admit(
        &plan,
        StatisticalAggregateExecutionMode::default(),
        PlanningPolicy::default(),
        rapids_analysis_context(),
    )
    .expect("scan/filter/aggregate admission should not error");
    let admission = decision.admission().expect("scan/filter/aggregate uses compiled admission");
    let pipeline = admission.pipeline().clone();
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
        let metrics = ExecutionMetrics::detached("aggregate-lane-parity");
        let metrics_context = metrics.scope_context(&test_context).expect("metrics scope applies");
        let gpu_inputs = GpuResidentInputMap::new();
        let scheduler = CompiledPipelineScheduler::new(&pipeline).expect("aggregate pipeline is schedulable");
        let mut executor =
            CompiledGpuTaskExecutor::new(&metrics_context, &plan, admission.policy(), &pipeline, &gpu_inputs, &metrics)
                .unwrap();
        let mut support = LaneDispatchSupport::new(
            Arc::clone(&route),
            metrics_context.clone(),
            plan.clone(),
            admission.policy(),
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
    let resident_pipeline = admit(
        &resident_plan,
        StatisticalAggregateExecutionMode::default(),
        PlanningPolicy::default(),
        rapids_analysis_context(),
    )
    .expect("resident aggregate admission should not error")
    .admission()
    .expect("resident aggregate uses compiled admission")
    .pipeline()
    .clone();
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
    let grouped_decision = admit(
        &grouped_plan,
        StatisticalAggregateExecutionMode::default(),
        PlanningPolicy::default(),
        rapids_analysis_context(),
    )
    .expect("grouped aggregate admission should not error");
    assert!(grouped_decision.admission().is_some(), "grouped aggregate uses compiled admission");
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
    let decision = admit(
        &plan,
        StatisticalAggregateExecutionMode::default(),
        PlanningPolicy::default(),
        rapids_analysis_context(),
    )
    .expect("overflow aggregate admission should not error");
    let admission = decision.admission().expect("overflow aggregate uses compiled admission");
    let pipeline = admission.pipeline().clone();
    assert!(is_lane_family_pipeline(&pipeline));

    let runtime = lane_shared_runtime(runtime);
    let test_context = runtime.acquire_context_for_plan(&plan).expect("overflow aggregate acquires context");
    let route = test_context.attempt_stream_provider_route().expect("overflow aggregate has a fork route");
    assert!(route.lane_capacity() >= 4);
    let run = |width: usize| -> std::result::Result<Vec<Option<i64>>, String> {
        let metrics = ExecutionMetrics::detached("aggregate-overflow-lane-parity");
        let metrics_context = metrics.scope_context(&test_context).expect("metrics scope applies");
        let gpu_inputs = GpuResidentInputMap::new();
        let scheduler = CompiledPipelineScheduler::new(&pipeline).expect("overflow pipeline is schedulable");
        let mut executor =
            CompiledGpuTaskExecutor::new(&metrics_context, &plan, admission.policy(), &pipeline, &gpu_inputs, &metrics)
                .unwrap();
        let support = LaneDispatchSupport::new(
            Arc::clone(&route),
            metrics_context.clone(),
            plan.clone(),
            admission.policy(),
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

/// Join DAG branches overlap on lanes while one build serves every probe.
///
/// The probe side is a union of two literal branches; the build side is a
/// third literal. Width 1 runs the lane protocol inline, width 2 runs the
/// two union branches on pool workers with disjoint forks, and the serial
/// scheduler is the baseline. All three share `execute_task` and the
/// claim/output-ready/commit protocol. The hash table is built once
/// (`hash_join_build_count == 1`) regardless of probe branch count, and the
/// lane gate proves the two branches occupied distinct lanes and forks.
#[test]
fn join_lane_dispatch_overlaps_branches_and_reuses_one_build() {
    use std::sync::Mutex;

    let runtime = cudf::Runtime::new().expect("compiled GPU execution test requires a CUDA runtime");
    let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false)]));
    let build_batch =
        RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef])
            .expect("join build literal batch");
    let left_batch =
        RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef])
            .expect("join probe left literal batch");
    let right_batch =
        RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from(vec![2, 3])) as ArrayRef])
            .expect("join probe right literal batch");
    let mut builder = QueryPlanBuilder::new();
    let build =
        builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(Arc::clone(&schema), vec![build_batch])));
    let left =
        builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(Arc::clone(&schema), vec![left_batch])));
    let right =
        builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(Arc::clone(&schema), vec![right_batch])));
    let probe = builder.add_union(vec![left, right], Arc::clone(&schema));
    let join = builder.add_join(
        build,
        probe,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "k", "k"), ColumnRef::input(1, "k", "k"))],
        )
        .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "k", "k"), "k")])
        .with_build_side(JoinSide::Left),
        Arc::clone(&schema),
    );
    let plan = builder.finish_with_host_sink(join);

    let decision = admit(
        &plan,
        StatisticalAggregateExecutionMode::default(),
        PlanningPolicy::default(),
        rapids_analysis_context(),
    )
    .expect("bounded admission should not error");
    let admission = decision.admission().expect("join branches use the compiled admission");
    let pipeline = admission.pipeline().clone();
    assert!(is_lane_family_pipeline(&pipeline), "join pipelines dispatch on lanes");
    assert!(
        pipeline.tasks().iter().any(|task| task.kind() == PipelineTaskKind::JoinBuild),
        "the join trio stays in the lane family on the coordinator"
    );

    let runtime = lane_shared_runtime(runtime);
    let test_context = runtime.acquire_context_for_plan(&plan).expect("join plan acquires a context");
    let route = test_context.attempt_stream_provider_route().expect("join attempt owns a fork route");

    let run = |requested_lanes: Option<usize>, gate: bool| {
        let metrics = ExecutionMetrics::detached("join-lane-parity");
        let metrics_context = metrics.scope_context(&test_context).expect("metrics scope applies");
        let gpu_inputs = GpuResidentInputMap::new();
        let scheduler = CompiledPipelineScheduler::new(&pipeline).expect("join pipeline is schedulable");
        let mut executor =
            CompiledGpuTaskExecutor::new(&metrics_context, &plan, admission.policy(), &pipeline, &gpu_inputs, &metrics)
                .unwrap();
        let observed = Arc::new((Mutex::new(Vec::new()), Condvar::new()));
        let completed = match requested_lanes {
            None => {
                let state = scheduler.run(&mut executor).expect("serial join should execute");
                assert_eq!(state.live_resource_count(), 0, "serial run releases every resource");
                executor.take_host_output().expect("serial join materializes")
            },
            Some(width) => {
                let mut support = LaneDispatchSupport::new(
                    Arc::clone(&route),
                    metrics_context.clone(),
                    plan.clone(),
                    admission.policy(),
                    pipeline.clone(),
                    gpu_inputs.clone(),
                    metrics.clone(),
                )
                .with_requested_lanes(width);
                if gate {
                    // Rendezvous gate: both lane workers must reach completion
                    // (holding uncommitted outputs) before either returns.
                    // Sequential execution on distinct forks cannot pass: the
                    // first worker blocks in the gate until the second
                    // arrives, proving simultaneous in-flight branch work and
                    // distinct fork ownership, not just post-hoc IDs.
                    let observed = Arc::clone(&observed);
                    support = support.with_lane_completion_gate(Arc::new(move |lane, fork| {
                        let (mutex, wake) = &*observed;
                        let mut guard = mutex.lock().expect("lane observation mutex");
                        guard.push((lane, fork));
                        wake.notify_all();
                        let (guard, _) = wake
                            .wait_timeout_while(guard, Duration::from_secs(10), |guard| guard.len() < 2)
                            .expect("lane rendezvous wait");
                        assert!(guard.len() >= 2, "two branch lanes must hold uncommitted outputs simultaneously");
                    }));
                }
                let state = run_lane_dispatched(&scheduler, &mut executor, None, &support)
                    .expect("lane-dispatched join should execute");
                assert_eq!(state.live_resource_count(), 0, "lane run releases every resource");
                executor.take_host_output().expect("lane join materializes")
            },
        };
        let mut values = int32_values(&completed, 0);
        values.sort();
        let builds = metrics.snapshot().hash_join_build_count;
        let observed = observed.0.lock().expect("observation mutex").clone();
        (values, builds, observed)
    };

    let (expected, expected_builds, _) = run(None, false);
    assert_eq!(expected, vec![Some(1), Some(2), Some(2)], "serial baseline joins every probe row once");
    assert_eq!(expected_builds, 1, "serial build runs once for every probe branch");
    let (width1, width1_builds, _) = run(Some(1), false);
    assert_eq!(width1, expected, "width 1 runs the same lane protocol inline");
    assert_eq!(width1_builds, 1, "width 1 still builds once");
    let (width2, width2_builds, observed) = run(Some(2), true);
    assert_eq!(width2, expected, "width 2 computes every branch input exactly once");
    assert_eq!(width2_builds, 1, "probe branch count never grows the build count");
    let mut lanes: Vec<_> = observed.iter().map(|(lane, _)| *lane).collect();
    lanes.sort();
    lanes.dedup();
    assert!(lanes.len() >= 2, "two union branches occupied distinct lanes: {observed:?}");
    let mut forks: Vec<_> = observed.iter().filter_map(|(_, fork)| *fork).collect();
    forks.sort();
    forks.dedup();
    assert!(forks.len() >= 2, "two union branches owned distinct fork streams: {observed:?}");
}

/// Probe batch growth never grows the build count, across join kinds.
///
/// The probe parquet doubles its row groups while the build stays one
/// literal; both shapes report exactly one hash build and identical joined
/// rows. Inner and left-semi stand in for the parallel-safe chunk contract
/// (concatenation preserves membership); the serial outer path is covered by
/// width parity so a future parallel outer merge cannot drift semantics.
#[test]
fn join_lane_dispatch_probe_batches_scale_without_rebuilding() {
    let dir = unique_temp_dir("query-engine-join-probe-scaling");
    let build_schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false)]));
    let probe_schema =
        Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false), Field::new("v", DataType::Int64, false)]));
    let run_shape = |probe_rows: Vec<i32>, join_type: JoinType| {
        let path = dir.join(format!("probe-{}-{}.parquet", probe_rows.len(), join_type));
        let values: Vec<i64> = (0..probe_rows.len() as i64).collect();
        let batch = RecordBatch::try_new(
            Arc::clone(&probe_schema),
            vec![Arc::new(Int32Array::from(probe_rows)) as ArrayRef, Arc::new(Int64Array::from(values)) as ArrayRef],
        )
        .expect("probe batch");
        let props = WriterProperties::builder().set_max_row_group_row_count(Some(2)).build();
        let mut writer = ArrowWriter::try_new(File::create(&path).unwrap(), Arc::clone(&probe_schema), Some(props))
            .expect("probe parquet writer");
        writer.write(&batch).expect("probe parquet write");
        writer.close().expect("probe parquet close");
        let build_batch =
            RecordBatch::try_new(Arc::clone(&build_schema), vec![Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef])
                .expect("build batch");
        let mut builder = QueryPlanBuilder::new();
        let build = builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(
            Arc::clone(&build_schema),
            vec![build_batch],
        )));
        let probe = builder.add_source(SourceNode::LocalParquet(fixture_local_parquet_source(&path, &probe_schema)));
        // Filtered semi builds on the right dimension side; inner builds on
        // the left literal. Both keep one build for every probe batch.
        let (output_schema, projection, build_side, left, right) = match join_type {
            JoinType::LeftSemi => (
                Arc::new(Schema::new(vec![
                    Field::new("k", DataType::Int32, false),
                    Field::new("v", DataType::Int64, false),
                ])),
                vec![
                    JoinOutputColumn::left(ColumnRef::input(0, "k", "k"), "k"),
                    JoinOutputColumn::left(ColumnRef::input(0, "v", "v"), "v"),
                ],
                JoinSide::Right,
                probe,
                build,
            ),
            _ => (
                Arc::new(Schema::new(vec![
                    Field::new("k", DataType::Int32, false),
                    Field::new("v", DataType::Int64, false),
                ])),
                vec![
                    JoinOutputColumn::left(ColumnRef::input(0, "k", "k"), "k"),
                    JoinOutputColumn::right(ColumnRef::input(1, "v", "v"), "v"),
                ],
                JoinSide::Left,
                build,
                probe,
            ),
        };
        let join = builder.add_join(
            left,
            right,
            JoinSpec::new(join_type, vec![JoinKey::new(ColumnRef::input(0, "k", "k"), ColumnRef::input(1, "k", "k"))])
                .with_output_projection(projection)
                .with_build_side(build_side),
            output_schema,
        );
        let plan = builder.finish_with_host_sink(join);
        let decision = admit(
            &plan,
            StatisticalAggregateExecutionMode::default(),
            PlanningPolicy::default(),
            rapids_analysis_context(),
        )
        .expect("join admission should not error");
        let admission = decision.admission().unwrap_or_else(|| panic!("join {join_type} uses compiled admission"));
        let pipeline = admission.pipeline().clone();
        let runtime = lane_shared_runtime(cudf::Runtime::new().expect("join scaling runtime"));
        let test_context = runtime.acquire_context_for_plan(&plan).expect("join scaling context");
        let route = test_context.attempt_stream_provider_route().expect("join scaling route");
        let run = |width: usize| {
            let metrics = ExecutionMetrics::detached("join-probe-scaling");
            let metrics_context = metrics.scope_context(&test_context).expect("metrics scope applies");
            let gpu_inputs = GpuResidentInputMap::new();
            let scheduler = CompiledPipelineScheduler::new(&pipeline).expect("join scaling schedulable");
            let mut executor = CompiledGpuTaskExecutor::new(
                &metrics_context,
                &plan,
                admission.policy(),
                &pipeline,
                &gpu_inputs,
                &metrics,
            )
            .unwrap();
            let support = LaneDispatchSupport::new(
                Arc::clone(&route),
                metrics_context.clone(),
                plan.clone(),
                admission.policy(),
                pipeline.clone(),
                gpu_inputs.clone(),
                metrics.clone(),
            )
            .with_requested_lanes(width);
            let state = run_lane_dispatched(&scheduler, &mut executor, None, &support)
                .expect("join scaling lanes should execute");
            assert_eq!(state.live_resource_count(), 0);
            let batch = executor.take_host_output().expect("join scaling output");
            let snapshot = metrics.snapshot();
            (batch, snapshot.hash_join_build_count, snapshot.filtered_join_build_count)
        };
        (run(1), run(2))
    };
    let ((inner_batch1, builds1, _), (inner_batch2, builds2, _)) = run_shape(vec![1, 2], JoinType::Inner);
    let small1 = int32_int64_pairs_sorted(&inner_batch1, 0, 1);
    assert_eq!((builds1, builds2), (1, 1), "small probe builds once on both widths");
    assert_eq!(int32_int64_pairs_sorted(&inner_batch2, 0, 1), small1);
    assert_eq!(small1, vec![(Some(1), Some(0)), (Some(2), Some(1))]);
    let ((inner_batch1, builds1, _), (inner_batch2, builds2, _)) =
        run_shape(vec![1, 2, 2, 3, 1, 2, 2, 3], JoinType::Inner);
    let inner1 = int32_int64_pairs_sorted(&inner_batch1, 0, 1);
    let inner2 = int32_int64_pairs_sorted(&inner_batch2, 0, 1);
    assert_eq!(builds1, 1, "large probe still builds once (width 1)");
    assert_eq!(builds2, 1, "large probe still builds once (width 2): batch growth never grows builds");
    assert_eq!(inner1, inner2, "widths agree on large inner probe batches");
    assert_eq!(inner1.len(), small1.len() * 2 + 2, "large probe joins every row once");
    let ((semi_batch1, semi_hash1, semi_f1), (semi_batch2, semi_hash2, semi_f2)) =
        run_shape(vec![1, 2, 2, 3, 1, 2, 2, 3], JoinType::LeftSemi);
    let semi1 = int32_int64_pairs_sorted(&semi_batch1, 0, 1);
    let semi2 = int32_int64_pairs_sorted(&semi_batch2, 0, 1);
    assert_eq!((semi_hash1, semi_hash2), (0, 0), "filtered semi uses no hash build");
    assert_eq!((semi_f1, semi_f2), (1, 1), "filtered build runs once for every probe batch count");
    assert_eq!(semi1, semi2, "widths agree on semi probe batches");
    assert_eq!(semi1, inner1, "semi keeps every matching probe row once");
}

fn bool_values(batch: &RecordBatch, column: usize) -> Vec<Option<bool>> {
    use arrow::array::Array as _;
    let array = batch.column(column).as_any().downcast_ref::<BooleanArray>().expect("boolean column");
    (0..array.len()).map(|index| if array.is_null(index) { None } else { Some(array.value(index)) }).collect()
}

fn int32_strlens_sorted(batch: &RecordBatch) -> Vec<(Option<i32>, usize)> {
    use arrow::array::Array as _;
    let strings = batch.column(1).as_any().downcast_ref::<StringArray>().expect("utf8 payload column");
    let mut pairs = int32_values(batch, 0)
        .into_iter()
        .zip((0..strings.len()).map(|index| {
            assert!(!strings.is_null(index), "payload strings stay non-null");
            strings.value(index).len()
        }))
        .collect::<Vec<_>>();
    pairs.sort();
    pairs
}

fn int32_int64_int64_triples_sorted(
    batch: &RecordBatch,
    c0: usize,
    c1: usize,
    c2: usize,
) -> Vec<(Option<i32>, Option<i64>, Option<i64>)> {
    let mut triples = int32_values(batch, c0)
        .into_iter()
        .zip(int64_values(batch, c1))
        .zip(int64_values(batch, c2))
        .map(|((a, b), c)| (a, b, c))
        .collect::<Vec<_>>();
    triples.sort();
    triples
}

fn int32_bool_pairs_sorted(batch: &RecordBatch) -> Vec<(Option<i32>, Option<bool>)> {
    let mut pairs = int32_values(batch, 0).into_iter().zip(bool_values(batch, 1)).collect::<Vec<_>>();
    pairs.sort();
    pairs
}

fn int32_strlen_int64_triples_sorted(batch: &RecordBatch) -> Vec<(Option<i32>, usize, Option<i64>)> {
    use arrow::array::Array as _;
    let strings = batch.column(1).as_any().downcast_ref::<StringArray>().expect("utf8 payload column");
    let mut triples = int32_values(batch, 0)
        .into_iter()
        .zip((0..strings.len()).map(|index| strings.value(index).len()))
        .zip(int64_values(batch, 2))
        .map(|((a, b), c)| (a, b, c))
        .collect::<Vec<_>>();
    triples.sort();
    triples
}

/// A filtered semi over four leaf files feeds an eager four-frame stream
/// into a second hash join, whose probe then shares one build across chunks
/// on lanes. Wide payloads (~700KB per row) force the byte-driven chunk
/// planner to keep every file in its own chunk; tiny rows would be merged
/// into one chunk and collapse the stream to a single frame. The
/// `parallel_probe=hash` strategy decision proves the new path ran (chunk
/// count and lane width recorded); the hash/filtered build counts prove batch
/// growth never rebuilds; serial/width-1/width-2 agreement plus the
/// hand-checked rows prove semantics.
#[test]
fn chained_semi_feeds_hash_probe_chunks_in_parallel() {
    let dir = unique_temp_dir("query-engine-chained-hash-parallel-probe");
    // Wide payloads (~700KB per row) force the byte-driven chunk planner to
    // keep every probe file in its own chunk; tiny rows would be merged into
    // one chunk and collapse the stream to a single frame.
    let wide = "x".repeat(700_000);
    let probe_schema =
        Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false), Field::new("v", DataType::Utf8, false)]));
    let mut probe_files = Vec::new();
    for (file_index, keys) in [[1, 2], [2, 3], [1, 2], [2, 3]].into_iter().enumerate() {
        let path = dir.join(format!("probe-{file_index}.parquet"));
        let values = vec![wide.clone(), wide.clone()];
        let batch = RecordBatch::try_new(
            Arc::clone(&probe_schema),
            vec![
                Arc::new(Int32Array::from(keys.to_vec())) as ArrayRef,
                Arc::new(StringArray::from(values)) as ArrayRef,
            ],
        )
        .expect("chained probe batch");
        let mut writer = ArrowWriter::try_new(File::create(&path).unwrap(), Arc::clone(&probe_schema), None)
            .expect("chained probe writer");
        writer.write(&batch).expect("chained probe write");
        writer.close().expect("chained probe close");
        probe_files.push(enrich_fixture_local_parquet_file(LocalParquetFile::new(&path)));
    }
    let key_schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false)]));
    let kv_schema =
        Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false), Field::new("v", DataType::Utf8, false)]));
    let mut builder = QueryPlanBuilder::new();
    let b1 = builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(
        Arc::clone(&key_schema),
        vec![
            RecordBatch::try_new(Arc::clone(&key_schema), vec![Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef])
                .expect("semi build batch"),
        ],
    )));
    let p1 = builder.add_source(SourceNode::LocalParquet(NativeLocalParquetSource::from_files(
        probe_files,
        Arc::clone(&probe_schema),
        vec!["k".to_string(), "v".to_string()],
    )));
    let join1 = builder.add_join(
        p1,
        b1,
        JoinSpec::new(
            JoinType::LeftSemi,
            vec![JoinKey::new(ColumnRef::input(0, "k", "k"), ColumnRef::input(1, "k", "k"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "k", "k"), "k"),
            JoinOutputColumn::left(ColumnRef::input(0, "v", "v"), "v"),
        ])
        .with_build_side(JoinSide::Right),
        Arc::clone(&kv_schema),
    );
    let b2 = builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(
        Arc::clone(&key_schema),
        vec![
            RecordBatch::try_new(Arc::clone(&key_schema), vec![Arc::new(Int32Array::from(vec![2])) as ArrayRef])
                .expect("hash build batch"),
        ],
    )));
    let join2 = builder.add_join(
        join1,
        b2,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "k", "k"), ColumnRef::input(1, "k", "k"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "k", "k"), "k"),
            JoinOutputColumn::left(ColumnRef::input(0, "v", "v"), "v"),
        ])
        .with_build_side(JoinSide::Right),
        Arc::clone(&kv_schema),
    );
    let plan = builder.finish_with_host_sink(join2);
    let decision = admit(
        &plan,
        StatisticalAggregateExecutionMode::default(),
        PlanningPolicy::default(),
        rapids_analysis_context(),
    )
    .expect("chained join admission should not error");
    let admission = decision.admission().expect("chained joins use compiled admission");
    let pipeline = admission.pipeline().clone();
    assert_eq!(
        pipeline.tasks().iter().filter(|task| task.kind() == PipelineTaskKind::JoinBuild).count(),
        2,
        "two joins build separately"
    );
    let runtime = lane_shared_runtime(cudf::Runtime::new().expect("chained join runtime"));
    let test_context = runtime.acquire_context_for_plan(&plan).expect("chained join context");
    let route = test_context.attempt_stream_provider_route().expect("chained join route");
    let run = |width: Option<usize>| {
        let metrics = ExecutionMetrics::detached("chained-hash-parallel");
        let metrics_context = metrics.scope_context(&test_context).expect("metrics scope applies");
        let gpu_inputs = GpuResidentInputMap::new();
        let scheduler = CompiledPipelineScheduler::new(&pipeline).expect("chained pipeline is schedulable");
        let mut executor =
            CompiledGpuTaskExecutor::new(&metrics_context, &plan, admission.policy(), &pipeline, &gpu_inputs, &metrics)
                .unwrap();
        let batch = match width {
            None => {
                let state = scheduler.run(&mut executor).expect("serial chained joins should execute");
                assert_eq!(state.live_resource_count(), 0);
                executor.take_host_output().expect("serial chained output")
            },
            Some(width) => {
                let support = LaneDispatchSupport::new(
                    Arc::clone(&route),
                    metrics_context.clone(),
                    plan.clone(),
                    admission.policy(),
                    pipeline.clone(),
                    gpu_inputs.clone(),
                    metrics.clone(),
                )
                .with_requested_lanes(width);
                let state = run_lane_dispatched(&scheduler, &mut executor, None, &support)
                    .expect("lane chained joins should execute");
                assert_eq!(state.live_resource_count(), 0);
                executor.take_host_output().expect("lane chained output")
            },
        };
        (batch, metrics.snapshot())
    };
    let wide_len = "x".repeat(700_000).len();
    let expected = vec![(Some(2), wide_len), (Some(2), wide_len), (Some(2), wide_len), (Some(2), wide_len)];
    for (label, (batch, snapshot)) in [("serial", run(None)), ("width1", run(Some(1))), ("width2", run(Some(2)))] {
        assert_eq!(int32_strlens_sorted(&batch), expected, "{label} joins every chained probe row once");
        assert_eq!(snapshot.hash_join_build_count, 1, "{label} builds the hash table once");
        assert_eq!(snapshot.filtered_join_build_count, 1, "{label} builds the filtered table once");
        assert_eq!(
            snapshot.source_chunks_committed, 4,
            "{label} reads four probe files as four chunks feeding four probe batches"
        );
    }
    let (_, width2_snapshot) = run(Some(2));
    let parallel = width2_snapshot
        .join_strategy_decisions
        .iter()
        .find(|decision| decision.facts.get("parallel_probe").map(String::as_str) == Some("hash"))
        .expect("width-2 run records the hash parallel-probe decision");
    assert_eq!(
        parallel.facts.get("parallel_probe_chunks").map(String::as_str),
        Some("4"),
        "four probe chunks share the second join build"
    );
    assert!(
        parallel.facts.get("parallel_probe_width").map(String::as_str).unwrap_or("1").parse::<usize>().unwrap() >= 2,
        "probe chunks spread over at least two lanes: {:?}",
        parallel.facts
    );
}

/// Same chain shape with a faked-large build so the second join plans a
/// split partitioned build; its lazy stream then drains probe chunks in
/// parallel sharing the partitions. Skew handling stays inside the existing
/// per-chunk driver (split/retry/typed failure), which lanes reuse
/// unchanged; adding lanes never assumes the max partition shrinks.
#[test]
fn chained_semi_feeds_partitioned_probe_chunks_in_parallel() {
    let dir = unique_temp_dir("query-engine-chained-partitioned-parallel-probe");
    let wide = "x".repeat(700_000);
    let probe_schema =
        Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false), Field::new("v", DataType::Utf8, false)]));
    let mut probe_files = Vec::new();
    for (file_index, keys) in [[1, 2], [2, 3], [1, 2], [2, 3]].into_iter().enumerate() {
        let probe_path = dir.join(format!("probe-{file_index}.parquet"));
        let values = vec![wide.clone(), wide.clone()];
        let batch = RecordBatch::try_new(
            Arc::clone(&probe_schema),
            vec![
                Arc::new(Int32Array::from(keys.to_vec())) as ArrayRef,
                Arc::new(StringArray::from(values)) as ArrayRef,
            ],
        )
        .expect("partitioned chain probe batch");
        let mut writer = ArrowWriter::try_new(File::create(&probe_path).unwrap(), Arc::clone(&probe_schema), None)
            .expect("partitioned chain probe writer");
        writer.write(&batch).expect("partitioned chain probe write");
        writer.close().expect("partitioned chain probe close");
        probe_files.push(enrich_fixture_local_parquet_file(LocalParquetFile::new(&probe_path)));
    }
    let key_schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false)]));
    let kv_schema =
        Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false), Field::new("v", DataType::Utf8, false)]));
    let kvw_schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("v", DataType::Utf8, false),
        Field::new("w", DataType::Int64, false),
    ]));
    let build_schema =
        Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false), Field::new("w", DataType::Int64, false)]));
    let build_path = dir.join("build.parquet");
    let build_batch = RecordBatch::try_new(
        Arc::clone(&build_schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef,
            Arc::new(Int64Array::from(vec![100, 200])) as ArrayRef,
        ],
    )
    .expect("partitioned chain build batch");
    let mut build_writer = ArrowWriter::try_new(File::create(&build_path).unwrap(), Arc::clone(&build_schema), None)
        .expect("partitioned chain build writer");
    build_writer.write(&build_batch).expect("partitioned chain build write");
    build_writer.close().expect("partitioned chain build close");
    let mut builder = QueryPlanBuilder::new();
    let b1 = builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(
        Arc::clone(&key_schema),
        vec![
            RecordBatch::try_new(Arc::clone(&key_schema), vec![Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef])
                .expect("semi build batch"),
        ],
    )));
    let p1 = builder.add_source(SourceNode::LocalParquet(NativeLocalParquetSource::from_files(
        probe_files,
        Arc::clone(&probe_schema),
        vec!["k".to_string(), "v".to_string()],
    )));
    let join1 = builder.add_join(
        p1,
        b1,
        JoinSpec::new(
            JoinType::LeftSemi,
            vec![JoinKey::new(ColumnRef::input(0, "k", "k"), ColumnRef::input(1, "k", "k"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "k", "k"), "k"),
            JoinOutputColumn::left(ColumnRef::input(0, "v", "v"), "v"),
        ])
        .with_build_side(JoinSide::Right),
        Arc::clone(&kv_schema),
    );
    let mut build_file = enrich_fixture_local_parquet_file(LocalParquetFile::new(&build_path));
    build_file.row_group_rows = Some(vec![100_000_000]);
    let b2 = builder.add_source(SourceNode::LocalParquet(NativeLocalParquetSource::from_files(
        vec![build_file],
        Arc::clone(&build_schema),
        vec!["k".to_string(), "w".to_string()],
    )));
    let join2 = builder.add_join(
        join1,
        b2,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "k", "k"), ColumnRef::input(1, "k", "k"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "k", "k"), "k"),
            JoinOutputColumn::left(ColumnRef::input(0, "v", "v"), "v"),
            JoinOutputColumn::right(ColumnRef::input(1, "w", "w"), "w"),
        ])
        .with_build_side(JoinSide::Right),
        Arc::clone(&kvw_schema),
    );
    let plan = builder.finish_with_host_sink(join2);
    let decision = admit(
        &plan,
        StatisticalAggregateExecutionMode::default(),
        PlanningPolicy::default(),
        rapids_analysis_context(),
    )
    .expect("partitioned chain admission should not error");
    let admission = decision.admission().expect("partitioned chain uses compiled admission");
    let pipeline = admission.pipeline().clone();
    let runtime = lane_shared_runtime(cudf::Runtime::new().expect("partitioned chain runtime"));
    let test_context = runtime.acquire_context_for_plan(&plan).expect("partitioned chain context");
    let route = test_context.attempt_stream_provider_route().expect("partitioned chain route");
    let run = |width: Option<usize>| {
        let metrics = ExecutionMetrics::detached("chained-partitioned-parallel");
        let metrics_context = metrics.scope_context(&test_context).expect("metrics scope applies");
        let gpu_inputs = GpuResidentInputMap::new();
        let scheduler = CompiledPipelineScheduler::new(&pipeline).expect("partitioned pipeline is schedulable");
        let mut executor =
            CompiledGpuTaskExecutor::new(&metrics_context, &plan, admission.policy(), &pipeline, &gpu_inputs, &metrics)
                .unwrap();
        let batch = match width {
            None => {
                let state = scheduler.run(&mut executor).expect("serial partitioned chain should execute");
                assert_eq!(state.live_resource_count(), 0);
                executor.take_host_output().expect("serial partitioned output")
            },
            Some(width) => {
                let support = LaneDispatchSupport::new(
                    Arc::clone(&route),
                    metrics_context.clone(),
                    plan.clone(),
                    admission.policy(),
                    pipeline.clone(),
                    gpu_inputs.clone(),
                    metrics.clone(),
                )
                .with_requested_lanes(width);
                let state = run_lane_dispatched(&scheduler, &mut executor, None, &support)
                    .expect("lane partitioned chain should execute");
                assert_eq!(state.live_resource_count(), 0);
                executor.take_host_output().expect("lane partitioned output")
            },
        };
        (batch, metrics.snapshot())
    };
    let wide_len = "x".repeat(700_000).len();
    let expected = vec![
        (Some(1), wide_len, Some(100)),
        (Some(1), wide_len, Some(100)),
        (Some(2), wide_len, Some(200)),
        (Some(2), wide_len, Some(200)),
        (Some(2), wide_len, Some(200)),
        (Some(2), wide_len, Some(200)),
    ];
    for (label, (batch, snapshot)) in [("serial", run(None)), ("width1", run(Some(1))), ("width2", run(Some(2)))] {
        assert_eq!(
            int32_strlen_int64_triples_sorted(&batch),
            expected,
            "{label} joins every partitioned probe row once"
        );
        assert_eq!(snapshot.hash_join_build_count, 0, "{label} builds no hash table");
        assert_eq!(
            snapshot.source_chunks_committed, 5,
            "{label} reads four probe files plus one build file as five chunks"
        );
    }
    let (_, width2_snapshot) = run(Some(2));
    assert!(
        width2_snapshot.join_strategy_decisions.iter().any(|decision| {
            decision.facts.get("join_strategy").map(String::as_str) == Some("partitioned_hash_equi")
        }),
        "second join plans a split partitioned build"
    );
    let parallel = width2_snapshot
        .join_strategy_decisions
        .iter()
        .find(|decision| decision.facts.get("parallel_probe").map(String::as_str) == Some("partitioned"))
        .expect("width-2 run records the partitioned parallel-probe decision");
    assert_eq!(
        parallel.facts.get("parallel_probe_chunks").map(String::as_str),
        Some("4"),
        "four probe chunks share the partitioned build"
    );
    assert!(
        parallel.facts.get("parallel_probe_width").map(String::as_str).unwrap_or("1").parse::<usize>().unwrap() >= 2,
        "partitioned probe chunks spread over at least two lanes: {:?}",
        parallel.facts
    );
}

/// Lane widths preserve every D-stage join contract: inner/left/semi/anti
/// with null keys and duplicate fanout, nulls-match equality, mark
/// membership, full-outer complements (serial fallback), build-left outer
/// (serial fallback), and residual predicates (serial fallback). Serial is
/// the CPU-oracle-checked baseline (existing bounded-join coverage proves
/// serial semantics); width-1/width-2 agreement proves lanes preserve them.
/// Hand-checked rows below pin the oracle for this data so a lane drift
/// cannot hide behind agreement alone.
#[test]
fn join_lane_widths_preserve_join_kinds_with_nulls_duplicates_and_residuals() {
    let key = || Field::new("k", DataType::Int32, true);
    let left_schema = Arc::new(Schema::new(vec![key(), Field::new("v", DataType::Int64, false)]));
    let right_schema =
        Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, true), Field::new("w", DataType::Int64, false)]));
    let left_batch = RecordBatch::try_new(
        Arc::clone(&left_schema),
        vec![
            Arc::new(Int32Array::from(vec![Some(1), None, Some(2), Some(2)])) as ArrayRef,
            Arc::new(Int64Array::from(vec![10, 20, 30, 40])) as ArrayRef,
        ],
    )
    .expect("null-key left batch");
    let right_batch = RecordBatch::try_new(
        Arc::clone(&right_schema),
        vec![
            Arc::new(Int32Array::from(vec![Some(2), None, Some(3)])) as ArrayRef,
            Arc::new(Int64Array::from(vec![200, 300, 400])) as ArrayRef,
        ],
    )
    .expect("null-key right batch");
    let kvw_schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, true),
        Field::new("v", DataType::Int64, true),
        Field::new("w", DataType::Int64, true),
    ]));
    let kv_schema =
        Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, true), Field::new("v", DataType::Int64, false)]));
    let mark_schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, true),
        Field::new("mark", DataType::Boolean, false),
    ]));

    fn execute3(plan: &QueryPlan, pipeline: &CompiledPipeline) -> (RecordBatch, RecordBatch, RecordBatch) {
        let runtime = lane_shared_runtime(cudf::Runtime::new().expect("parity runtime"));
        let test_context = runtime.acquire_context_for_plan(plan).expect("parity context");
        let route = test_context.attempt_stream_provider_route().expect("parity route");
        let metrics = ExecutionMetrics::detached("join-kind-parity");
        let metrics_context = metrics.scope_context(&test_context).expect("metrics scope applies");
        let gpu_inputs = GpuResidentInputMap::new();
        let scheduler = CompiledPipelineScheduler::new(pipeline).expect("parity pipeline is schedulable");
        let mut executor = CompiledGpuTaskExecutor::new(
            &metrics_context,
            plan,
            PlanningPolicy::default(),
            pipeline,
            &gpu_inputs,
            &metrics,
        )
        .unwrap();
        let mut run = |width: Option<usize>| match width {
            None => {
                let state = scheduler.run(&mut executor).expect("serial parity should execute");
                assert_eq!(state.live_resource_count(), 0);
                executor.take_host_output().expect("serial parity output")
            },
            Some(width) => {
                let support = LaneDispatchSupport::new(
                    Arc::clone(&route),
                    metrics_context.clone(),
                    plan.clone(),
                    PlanningPolicy::default(),
                    pipeline.clone(),
                    gpu_inputs.clone(),
                    metrics.clone(),
                )
                .with_requested_lanes(width);
                let state =
                    run_lane_dispatched(&scheduler, &mut executor, None, &support).expect("lane parity should execute");
                assert_eq!(state.live_resource_count(), 0);
                executor.take_host_output().expect("lane parity output")
            },
        };
        (run(None), run(Some(1)), run(Some(2)))
    }

    fn admitted_pipeline(label: &str, plan: &QueryPlan) -> CompiledPipeline {
        admit(plan, StatisticalAggregateExecutionMode::default(), PlanningPolicy::default(), rapids_analysis_context())
            .expect("parity admission should not error")
            .admission()
            .unwrap_or_else(|| panic!("{label}: parity shape uses compiled admission"))
            .pipeline()
            .clone()
    }

    // (label, join_type, build_side, null_equality, residual)
    // Build-left outer has no compiled strategy (structured fallback to the
    // generic executor, verified by existing admission coverage), so lane
    // preservation is vacuous there; full-outer below covers the serial
    // complement fallback inside compiled execution.
    for (label, join_type, build_side, null_eq, residual) in [
        ("inner", JoinType::Inner, JoinSide::Right, JoinNullEquality::NullsDoNotMatch, false),
        ("inner-nulls-match", JoinType::Inner, JoinSide::Right, JoinNullEquality::NullsMatch, false),
        ("left", JoinType::Left, JoinSide::Right, JoinNullEquality::NullsDoNotMatch, false),
        ("semi", JoinType::LeftSemi, JoinSide::Right, JoinNullEquality::NullsDoNotMatch, false),
        ("anti", JoinType::LeftAnti, JoinSide::Right, JoinNullEquality::NullsDoNotMatch, false),
        ("full", JoinType::Full, JoinSide::Right, JoinNullEquality::NullsDoNotMatch, false),
        ("inner-residual", JoinType::Inner, JoinSide::Right, JoinNullEquality::NullsDoNotMatch, true),
    ] {
        let mut builder = QueryPlanBuilder::new();
        let left = builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(
            Arc::clone(&left_schema),
            vec![left_batch.clone()],
        )));
        let right = builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(
            Arc::clone(&right_schema),
            vec![right_batch.clone()],
        )));
        let (projection, output_schema) = match join_type {
            JoinType::LeftSemi | JoinType::LeftAnti => (
                vec![
                    JoinOutputColumn::left(ColumnRef::input(0, "k", "k"), "k"),
                    JoinOutputColumn::left(ColumnRef::input(0, "v", "v"), "v"),
                ],
                Arc::clone(&kv_schema),
            ),
            _ => (
                vec![
                    JoinOutputColumn::left(ColumnRef::input(0, "k", "k"), "k"),
                    JoinOutputColumn::left(ColumnRef::input(0, "v", "v"), "v"),
                    JoinOutputColumn::right(ColumnRef::input(1, "w", "w"), "w"),
                ],
                Arc::clone(&kvw_schema),
            ),
        };
        let mut spec =
            JoinSpec::new(join_type, vec![JoinKey::new(ColumnRef::input(0, "k", "k"), ColumnRef::input(1, "k", "k"))])
                .with_output_projection(projection)
                .with_null_equality(null_eq)
                .with_build_side(build_side);
        if residual {
            spec = spec.with_residual_predicate(Expr::binary(
                Expr::input_column(0, "v", "v"),
                BinaryOp::Lt,
                Expr::input_column(1, "w", "w"),
            ));
        }
        let join = builder.add_join(left, right, spec, output_schema);
        let plan = builder.finish_with_host_sink(join);
        let pipeline = admitted_pipeline(label, &plan);
        let (serial, width1, width2) = execute3(&plan, &pipeline);
        assert_eq!(width1.num_rows(), serial.num_rows(), "{label}: width 1 keeps row count");
        assert_eq!(width2.num_rows(), serial.num_rows(), "{label}: width 2 keeps row count");
        match join_type {
            JoinType::LeftSemi | JoinType::LeftAnti => {
                let expected = int32_int64_pairs_sorted(&serial, 0, 1);
                assert_eq!(int32_int64_pairs_sorted(&width1, 0, 1), expected, "{label}: width 1 preserves rows");
                assert_eq!(int32_int64_pairs_sorted(&width2, 0, 1), expected, "{label}: width 2 preserves rows");
                let oracle = match join_type {
                    JoinType::LeftSemi => vec![(Some(2), Some(30)), (Some(2), Some(40))],
                    _ => vec![(None, Some(20)), (Some(1), Some(10))],
                };
                assert_eq!(expected, oracle, "{label}: serial matches the null-key oracle");
            },
            _ => {
                let expected = int32_int64_int64_triples_sorted(&serial, 0, 1, 2);
                assert_eq!(
                    int32_int64_int64_triples_sorted(&width1, 0, 1, 2),
                    expected,
                    "{label}: width 1 preserves rows"
                );
                assert_eq!(
                    int32_int64_int64_triples_sorted(&width2, 0, 1, 2),
                    expected,
                    "{label}: width 2 preserves rows"
                );
                let oracle = match (label, join_type) {
                    (_, JoinType::Inner) if !residual => {
                        vec![(Some(2), Some(30), Some(200)), (Some(2), Some(40), Some(200))]
                    },
                    (_, JoinType::Inner) => {
                        vec![(Some(2), Some(30), Some(200)), (Some(2), Some(40), Some(200))]
                    },
                    (_, JoinType::Left) => vec![
                        (None, Some(20), None),
                        (Some(1), Some(10), None),
                        (Some(2), Some(30), Some(200)),
                        (Some(2), Some(40), Some(200)),
                    ],
                    (_, JoinType::Full) => vec![
                        (None, None, Some(300)),
                        (None, None, Some(400)),
                        (None, Some(20), None),
                        (Some(1), Some(10), None),
                        (Some(2), Some(30), Some(200)),
                        (Some(2), Some(40), Some(200)),
                    ],
                    _ => unreachable!("oracle covers every parity case"),
                };
                // Nulls-match inner adds the (None,20,300) row; residual
                // keeps the same rows here because every matched pair
                // satisfies v < w.
                let mut oracle = oracle;
                if label == "inner-nulls-match" {
                    oracle.push((None, Some(20), Some(300)));
                    oracle.sort();
                }
                assert_eq!(expected, oracle, "{label}: serial matches the null-key oracle");
            },
        }
    }

    // Mark membership reports every probe row with its boolean flag.
    let mut builder = QueryPlanBuilder::new();
    let left = builder
        .add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(Arc::clone(&left_schema), vec![left_batch])));
    let right = builder
        .add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(Arc::clone(&right_schema), vec![right_batch])));
    let mark = builder.add_join(
        left,
        right,
        JoinSpec::new(
            JoinType::LeftMark,
            vec![JoinKey::new(ColumnRef::input(0, "k", "k"), ColumnRef::input(1, "k", "k"))],
        )
        .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "k", "k"), "k")])
        .with_null_equality(JoinNullEquality::NullsDoNotMatch)
        .with_build_side(JoinSide::Right)
        .with_mark_column_name("mark"),
        Arc::clone(&mark_schema),
    );
    let plan = builder.finish_with_host_sink(mark);
    let pipeline = admitted_pipeline("mark", &plan);
    let (serial, width1, width2) = execute3(&plan, &pipeline);
    let oracle = vec![(None, Some(false)), (Some(1), Some(false)), (Some(2), Some(true)), (Some(2), Some(true))];
    assert_eq!(int32_bool_pairs_sorted(&serial), oracle, "mark: serial matches the membership oracle");
    assert_eq!(int32_bool_pairs_sorted(&width1), oracle, "mark: width 1 preserves membership");
    assert_eq!(int32_bool_pairs_sorted(&width2), oracle, "mark: width 2 preserves membership");
}

/// Cancelling while join branch workers hold uncommitted outputs drains
/// without committing: the coordinator re-checks cancellation after joining
/// workers, releases every uncommitted output through accounting (including
/// coordinator-owned join state via the error-cleanup path), reports typed
/// `Cancelled`, and materializes no sink output. Late completions can never
/// resurrect because commit requires re-checking cancellation first.
#[test]
fn join_lane_cancellation_drains_branch_work_without_committing() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let runtime = cudf::Runtime::new().expect("compiled GPU execution test requires a CUDA runtime");
    let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false)]));
    let build_batch =
        RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef])
            .expect("cancel build batch");
    let left_batch =
        RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef])
            .expect("cancel left batch");
    let right_batch =
        RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from(vec![2, 3])) as ArrayRef])
            .expect("cancel right batch");
    let mut builder = QueryPlanBuilder::new();
    let build =
        builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(Arc::clone(&schema), vec![build_batch])));
    let left =
        builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(Arc::clone(&schema), vec![left_batch])));
    let right =
        builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(Arc::clone(&schema), vec![right_batch])));
    let probe = builder.add_union(vec![left, right], Arc::clone(&schema));
    let join = builder.add_join(
        build,
        probe,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "k", "k"), ColumnRef::input(1, "k", "k"))],
        )
        .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "k", "k"), "k")])
        .with_build_side(JoinSide::Left),
        Arc::clone(&schema),
    );
    let plan = builder.finish_with_host_sink(join);
    let decision = admit(
        &plan,
        StatisticalAggregateExecutionMode::default(),
        PlanningPolicy::default(),
        rapids_analysis_context(),
    )
    .expect("cancel admission should not error");
    let admission = decision.admission().expect("cancel join uses compiled admission");
    let pipeline = admission.pipeline().clone();
    let runtime = lane_shared_runtime(runtime);
    let test_context = runtime.acquire_context_for_plan(&plan).expect("cancel context");
    let route = test_context.attempt_stream_provider_route().expect("cancel route");
    assert!(route.lane_capacity() >= 2, "cancel test needs two branch lanes");
    let metrics = ExecutionMetrics::detached("join-lane-cancel");
    let metrics_context = metrics.scope_context(&test_context).expect("metrics scope applies");
    let gpu_inputs = GpuResidentInputMap::new();
    let scheduler = CompiledPipelineScheduler::new(&pipeline).expect("cancel pipeline is schedulable");
    let mut executor =
        CompiledGpuTaskExecutor::new(&metrics_context, &plan, admission.policy(), &pipeline, &gpu_inputs, &metrics)
            .unwrap();
    // Park the first two lane completions (holding uncommitted branch
    // outputs) until the canceller fires; later rounds are unaffected. The
    // Condvar timeouts turn a lost rendezvous into a test failure instead
    // of a hung worker.
    let entered = Arc::new((Mutex::new(0usize), Condvar::new()));
    let released = Arc::new((Mutex::new(false), Condvar::new()));
    let remaining = Arc::new(AtomicUsize::new(2));
    let gate_entered = Arc::clone(&entered);
    let gate_released = Arc::clone(&released);
    let support = LaneDispatchSupport::new(
        Arc::clone(&route),
        metrics_context.clone(),
        plan.clone(),
        admission.policy(),
        pipeline.clone(),
        gpu_inputs.clone(),
        metrics.clone(),
    )
    .with_requested_lanes(2)
    .with_lane_completion_gate(Arc::new(move |_, _| {
        // Only the first two completions park; later rounds skip the gate.
        if remaining.fetch_sub(1, Ordering::AcqRel) < 1 {
            return;
        }
        {
            let (mutex, wake) = &*gate_entered;
            *mutex.lock().expect("cancel gate mutex") += 1;
            wake.notify_all();
        }
        let (mutex, wake) = &*gate_released;
        let (guard, result) = wake
            .wait_timeout_while(mutex.lock().expect("cancel release mutex"), Duration::from_secs(10), |ready| !*ready)
            .expect("cancel release wait");
        assert!(*guard || result.timed_out(), "release gate always wakes parked workers");
    }));
    let cancellation = test_context.cancellation().clone();
    let waiter_entered = Arc::clone(&entered);
    let waiter_released = Arc::clone(&released);
    let canceller = std::thread::spawn(move || {
        let (mutex, wake) = &*waiter_entered;
        let (guard, result) = wake
            .wait_timeout_while(mutex.lock().expect("canceller mutex"), Duration::from_secs(10), |count| *count < 2)
            .expect("canceller wait");
        assert_eq!(*guard, 2, "both branch workers park holding uncommitted outputs");
        assert!(!result.timed_out(), "branch workers rendezvous before the timeout");
        cancellation.cancel();
        *waiter_released.0.lock().expect("release mutex") = true;
        waiter_released.1.notify_all();
    });
    let error = match run_lane_dispatched(&scheduler, &mut executor, None, &support) {
        Ok(_) => panic!("a cancelled join run must not commit its branch outputs"),
        Err(error) => error,
    };
    canceller.join().expect("the cancelling thread joins");
    assert_eq!(error.code(), QueryErrorCode::Cancelled, "cancellation is typed");
    assert!(executor.take_host_output().is_err(), "no sink output is committed on a cancelled join run");
}

/// Eager lane scheduling must not lose dynamic-filter pruning: the optimized
/// plan carries a build-complete-before-consumer route, so the probe source
/// waits for the coordinator build and reads with the filter snapshot.
/// Serial/width-1/width-2 must prune the same row groups, apply the filter,
/// and return the same rows. The all-`3` row group is pruned by the `[2,2]`
/// build range on every width.
#[test]
fn join_lane_widths_preserve_dynamic_filter_pruning() {
    let dir = unique_temp_dir("query-engine-join-lane-dynamic-filter-pruning");
    let probe_schema =
        Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false), Field::new("v", DataType::Int64, false)]));
    let probe_path = dir.join("probe.parquet");
    let batch = RecordBatch::try_new(
        Arc::clone(&probe_schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 1, 2, 2, 3, 3])) as ArrayRef,
            Arc::new(Int64Array::from(vec![0, 1, 2, 3, 4, 5])) as ArrayRef,
        ],
    )
    .expect("pruning probe batch");
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(2)).build();
    let mut writer = ArrowWriter::try_new(File::create(&probe_path).unwrap(), Arc::clone(&probe_schema), Some(props))
        .expect("pruning probe writer");
    writer.write(&batch).expect("pruning probe write");
    writer.close().expect("pruning probe close");
    let build_schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false)]));
    let mut builder = QueryPlanBuilder::new();
    let probe = builder.add_source(SourceNode::LocalParquet(fixture_local_parquet_source(&probe_path, &probe_schema)));
    let build_batch =
        RecordBatch::try_new(Arc::clone(&build_schema), vec![Arc::new(Int32Array::from(vec![2])) as ArrayRef])
            .expect("pruning build batch");
    let build = builder
        .add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(Arc::clone(&build_schema), vec![build_batch])));
    let output_schema =
        Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false), Field::new("v", DataType::Int64, false)]));
    let join = builder.add_join(
        probe,
        build,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "k", "k"), ColumnRef::input(1, "k", "k"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "k", "k"), "k"),
            JoinOutputColumn::left(ColumnRef::input(0, "v", "v"), "v"),
        ])
        .with_build_side(JoinSide::Right),
        Arc::clone(&output_schema),
    );
    let raw = builder.finish_with_host_sink(join);
    let optimized = optimize_query_plan(
        raw,
        &NativeOptimizerOptions::default(),
        rapids_analysis_context(),
        local_dimension_pruning_resolver(),
    );
    assert!(
        !optimized.plan.dynamic_filter_routes().is_empty(),
        "optimizer discovers a build-to-probe dynamic-filter route"
    );
    let plan = optimized.plan;
    let decision = admit(
        &plan,
        StatisticalAggregateExecutionMode::default(),
        PlanningPolicy::default(),
        rapids_analysis_context(),
    )
    .expect("pruning admission should not error");
    let admission = decision.admission().expect("pruning shape uses compiled admission");
    let pipeline = admission.pipeline().clone();
    let runtime = lane_shared_runtime(cudf::Runtime::new().expect("pruning runtime"));
    let test_context = runtime.acquire_context_for_plan(&plan).expect("pruning context");
    let route = test_context.attempt_stream_provider_route().expect("pruning route");
    let run = |width: Option<usize>| {
        let metrics = ExecutionMetrics::detached("join-lane-pruning");
        let metrics_context = metrics.scope_context(&test_context).expect("metrics scope applies");
        let gpu_inputs = GpuResidentInputMap::new();
        let scheduler = CompiledPipelineScheduler::new(&pipeline).expect("pruning pipeline is schedulable");
        let mut executor =
            CompiledGpuTaskExecutor::new(&metrics_context, &plan, admission.policy(), &pipeline, &gpu_inputs, &metrics)
                .unwrap();
        let batch = match width {
            None => {
                let state = scheduler.run(&mut executor).expect("serial pruning should execute");
                assert_eq!(state.live_resource_count(), 0);
                executor.take_host_output().expect("serial pruning output")
            },
            Some(width) => {
                let support = LaneDispatchSupport::new(
                    Arc::clone(&route),
                    metrics_context.clone(),
                    plan.clone(),
                    admission.policy(),
                    pipeline.clone(),
                    gpu_inputs.clone(),
                    metrics.clone(),
                )
                .with_requested_lanes(width);
                let state = run_lane_dispatched(&scheduler, &mut executor, None, &support)
                    .expect("lane pruning should execute");
                assert_eq!(state.live_resource_count(), 0);
                executor.take_host_output().expect("lane pruning output")
            },
        };
        (batch, metrics.snapshot())
    };
    let expected = vec![(Some(2), Some(2)), (Some(2), Some(3))];
    for (label, (batch, snapshot)) in [("serial", run(None)), ("width1", run(Some(1))), ("width2", run(Some(2)))] {
        assert_eq!(int32_int64_pairs_sorted(&batch, 0, 1), expected, "{label} joins pruned probe rows once");
        assert!(
            snapshot.dynamic_filter_runtime_rows.iter().any(|row| row.event == "applied"),
            "{label} applies the build filter on the probe source"
        );
        assert_eq!(
            snapshot.row_groups_committed, 1,
            "{label} prunes the all-1 and all-3 row groups instead of reading all three"
        );
    }
}

/// A GPU-resident sink over a join plus an aggregate still routes through the
/// resident lane path after the lane-family widening, and the join trio keeps
/// its serial coordinator contract there: `run_resident_aggregate_chunks`
/// runs the DAG on one coordinator executor (join state never crosses lanes;
/// only aggregate source chunks borrow lanes), and the sink adopts frames to
/// the root runtime with lane leases held through the handoff.
#[test]
fn resident_sink_join_with_aggregate_routes_through_resident_lanes() {
    let runtime = cudf::Runtime::new().expect("compiled GPU execution test requires a CUDA runtime");
    let dir = unique_temp_dir("query-engine-resident-join-lanes");
    let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false)]));
    let probe_schema =
        Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false), Field::new("v", DataType::Int64, false)]));
    let count_schema = Arc::new(Schema::new(vec![Field::new("rows", DataType::Int64, false)]));
    let build_batch =
        RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef])
            .expect("resident build batch");
    let probe_path = dir.join("probe.parquet");
    let probe_batch = RecordBatch::try_new(
        Arc::clone(&probe_schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 2, 3])) as ArrayRef,
            Arc::new(Int64Array::from(vec![10, 20, 30, 40])) as ArrayRef,
        ],
    )
    .expect("resident probe batch");
    let mut writer = ArrowWriter::try_new(File::create(&probe_path).unwrap(), Arc::clone(&probe_schema), None)
        .expect("resident probe writer");
    writer.write(&probe_batch).expect("resident probe write");
    writer.close().expect("resident probe close");
    let mut builder = QueryPlanBuilder::new();
    let build =
        builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(Arc::clone(&schema), vec![build_batch])));
    let probe = builder.add_source(SourceNode::LocalParquet(fixture_local_parquet_source(&probe_path, &probe_schema)));
    let join = builder.add_join(
        build,
        probe,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "k", "k"), ColumnRef::input(1, "k", "k"))],
        )
        .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "k", "k"), "k")])
        .with_build_side(JoinSide::Left),
        Arc::clone(&schema),
    );
    let aggregate = builder.add_aggregate(
        join,
        AggregateSpec::new(Vec::new(), vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("k"), "rows")]),
        Arc::clone(&count_schema),
    );
    let plan = builder.finish_gpu_resident(aggregate);
    let decision = admit(
        &plan,
        StatisticalAggregateExecutionMode::default(),
        PlanningPolicy::default(),
        rapids_analysis_context(),
    )
    .expect("resident admission should not error");
    assert!(decision.admission().is_some(), "resident join uses compiled admission");
    let runtime = lane_shared_runtime(runtime);
    let test_context = runtime.acquire_context_for_plan(&plan).expect("resident join acquires a context");
    let result = execute_query_plan(&test_context, &plan).expect("resident join lanes should execute");
    let QueryOutput::GpuResident(handle) = result.output() else {
        panic!("resident join retains its output");
    };
    assert_eq!(
        handle.frame().placement().bundle_id,
        handle.runtime().placement().bundle_id,
        "a resident join output is adopted onto the root runtime"
    );
    let handle = handle.clone();
    drop(result);
    let batch = materialize_gpu_result_to_host(handle, &mut QueryMetrics::default())
        .expect("resident join output stays readable after its lane leases are released");
    assert_eq!(int64_values(&batch, 0), vec![Some(3)], "inner join feeds three rows into the count");
}

/// A resident aggregate run that obtains no second lane executes every chunk on
/// the root context, so its sink returns the aggregate frame as produced. Root
/// adoption would copy the whole output device-to-device into the root runtime.
#[test]
fn resident_aggregate_without_lanes_skips_root_adoption_copy() {
    let runtime = cudf::Runtime::new().expect("compiled GPU execution test requires a CUDA runtime");
    let dir = unique_temp_dir("query-engine-resident-aggregate-no-lanes");
    let input_schema =
        Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false), Field::new("v", DataType::Int64, false)]));
    let path = dir.join("input.parquet");
    let batch = RecordBatch::try_new(
        Arc::clone(&input_schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 1, 2, 3])) as ArrayRef,
            Arc::new(Int64Array::from(vec![10, 20, 30, 40])) as ArrayRef,
        ],
    )
    .expect("resident aggregate input batch");
    let mut writer = ArrowWriter::try_new(File::create(&path).unwrap(), Arc::clone(&input_schema), None)
        .expect("resident aggregate parquet writer");
    writer.write(&batch).expect("resident aggregate parquet write");
    writer.close().expect("resident aggregate parquet close");
    let mut builder = QueryPlanBuilder::new();
    let scan = builder.add_source(SourceNode::LocalParquet(fixture_local_parquet_source(&path, &input_schema)));
    let aggregate = builder.add_aggregate(
        scan,
        AggregateSpec::new(Vec::new(), vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("v"), "total")]),
        Arc::new(Schema::new(vec![Field::new("total", DataType::Int64, true)])),
    );
    let plan = builder.finish_gpu_resident(aggregate);

    let runtime = lane_shared_runtime(runtime);
    let test_context = runtime.acquire_context_for_plan(&plan).expect("resident aggregate acquires a context");
    // Hold every auxiliary permit so the run obtains no lane.
    let mut held = Vec::new();
    while let Some(lease) =
        test_context.try_acquire_attempt_work_lease().expect("auxiliary try-acquire should not error")
    {
        held.push(lease);
    }
    let copies_before = device_copy::device_to_device_copy_count_for_test();
    let result = execute_query_plan(&test_context, &plan).expect("lane-less resident aggregate should execute");
    assert_eq!(
        device_copy::device_to_device_copy_count_for_test() - copies_before,
        0,
        "a lane-less resident sink must not be copied into the root runtime"
    );
    let QueryOutput::GpuResident(handle) = result.output() else {
        panic!("resident aggregate retains its output");
    };
    let handle = handle.clone();
    drop(result);
    drop(held);
    let batch = materialize_gpu_result_to_host(handle, &mut QueryMetrics::default())
        .expect("resident aggregate output stays readable");
    assert_eq!(int64_values(&batch, 0), vec![Some(100)]);
}

fn sort_lane_nullable_source(path: &Path, keys: &[Option<i32>], values: &[Option<i64>]) -> NativeLocalParquetSource {
    write_nullable_key_value_parquet(path, keys, values);
    let schema =
        Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, true), Field::new("v", DataType::Int64, true)]));
    NativeLocalParquetSource::from_files(
        vec![enrich_fixture_local_parquet_file(LocalParquetFile::new(path))],
        schema,
        vec!["k".to_string(), "v".to_string()],
    )
}

/// A fused top-k over a two-frame union stream sorts one candidate per
/// frame on disjoint lanes and merges on the coordinator. The rendezvous
/// gate proves both candidates are in execution simultaneously; the serial,
/// width-1, and width-2 outputs must agree exactly, including null-key
/// placement. Reverting the lane fan-out leaves the gate with a single
/// arrival and fails the overlap assertions.
#[test]
fn lane_dispatch_sort_topk_candidates_overlap_lanes_and_match_serial() {
    let runtime = cudf::Runtime::new().expect("compiled GPU execution test requires a CUDA runtime");
    let dir = unique_temp_dir("query-engine-lane-sort-topk");
    let schema =
        Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, true), Field::new("v", DataType::Int64, true)]));
    let left = sort_lane_nullable_source(
        &dir.join("left.parquet"),
        &[Some(5), Some(3), None, Some(1)],
        &[Some(50), Some(30), None, Some(10)],
    );
    let right = sort_lane_nullable_source(
        &dir.join("right.parquet"),
        &[Some(4), Some(2), Some(0)],
        &[Some(40), Some(20), Some(0)],
    );
    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(left));
    let right = builder.add_source(SourceNode::LocalParquet(right));
    let union = builder.add_union(vec![left, right], Arc::clone(&schema));
    let sort =
        builder.add_sort(union, SortSpec::new(vec![SortExpr::new(Expr::column("k"), true, true)]), Arc::clone(&schema));
    let limit = builder.add_limit(sort, LimitSpec::new(0, 6), Arc::clone(&schema));
    let plan = builder.finish_with_host_sink(limit);

    let decision = admit(
        &plan,
        StatisticalAggregateExecutionMode::default(),
        PlanningPolicy::default(),
        rapids_analysis_context(),
    )
    .expect("sort/top-k admission should not error");
    let admission = decision.admission().expect("union sort/top-k uses compiled admission");
    let pipeline = admission.pipeline().clone();
    assert!(is_lane_family_pipeline(&pipeline), "union sort/top-k dispatches on lanes");
    let sort_task = pipeline
        .tasks()
        .iter()
        .find(|task| task.kind() == PipelineTaskKind::FinalSort)
        .expect("union sort/top-k compiles a final sort task");
    assert!(
        matches!(sort_task.sort_strategy(), Some(SortStrategy::FusedTopK { .. })),
        "sort/limit fuses into a stream-capable top-k"
    );

    let runtime = lane_shared_runtime(runtime);
    let test_context = runtime.acquire_context_for_plan(&plan).expect("sort/top-k plan acquires a context");
    let route = test_context.attempt_stream_provider_route().expect("admitted sort owns a fork route");
    let arrivals = Arc::new((Mutex::new(Vec::new()), Condvar::new()));
    let run = |requested_lanes: Option<usize>, coordinate: bool| {
        let metrics = ExecutionMetrics::detached("sort-lane-parity");
        let metrics_context = metrics.scope_context(&test_context).expect("metrics scope applies");
        let gpu_inputs = GpuResidentInputMap::new();
        let scheduler = CompiledPipelineScheduler::new(&pipeline).expect("sort pipeline is schedulable");
        let mut executor =
            CompiledGpuTaskExecutor::new(&metrics_context, &plan, admission.policy(), &pipeline, &gpu_inputs, &metrics)
                .unwrap();
        if let Some(width) = requested_lanes {
            let mut support = LaneDispatchSupport::new(
                Arc::clone(&route),
                metrics_context.clone(),
                plan.clone(),
                admission.policy(),
                pipeline.clone(),
                gpu_inputs.clone(),
                metrics.clone(),
            )
            .with_requested_lanes(width);
            if coordinate {
                let arrivals = Arc::clone(&arrivals);
                support = support.with_aggregate_chunk_completion_gate(Arc::new(move |ordinal, lane, fork| {
                    let (mutex, wake) = &*arrivals;
                    let mut observed = mutex.lock().expect("sort arrival mutex");
                    observed.push((ordinal, lane, fork));
                    wake.notify_all();
                    let (observed, _) = wake
                        .wait_timeout_while(observed, Duration::from_secs(10), |observed| observed.len() < 2)
                        .expect("sort arrival wait");
                    assert!(observed.len() >= 2, "two top-k candidates must reach completion before either can return");
                }));
            }
            let state = run_lane_dispatched(&scheduler, &mut executor, None, &support)
                .expect("lane-dispatched sort/top-k should complete");
            assert_eq!(state.live_resource_count(), 0, "lane run releases every resource");
        } else {
            let state = scheduler.run(&mut executor).expect("serial sort/top-k should complete");
            assert_eq!(state.live_resource_count(), 0, "serial run releases every resource");
        }
        let batch = executor.take_host_output().expect("sort sink materializes one batch");
        int32_values(&batch, 0).into_iter().zip(int64_values(&batch, 1)).collect::<Vec<_>>()
    };

    let expected = vec![
        (None, None),
        (Some(5), Some(50)),
        (Some(4), Some(40)),
        (Some(3), Some(30)),
        (Some(2), Some(20)),
        (Some(1), Some(10)),
    ];
    assert_eq!(run(None, false), expected, "serial baseline orders nulls first, then keys descending");
    assert_eq!(run(Some(1), false), expected, "width 1 runs the same lane protocol inline");
    assert_eq!(run(Some(2), true), expected, "width 2 merges lane candidates into the same global top-k");
    let observed = arrivals.0.lock().expect("sort arrival mutex");
    assert!(observed.len() >= 2, "two top-k candidates must complete on lanes");
    assert_ne!(observed[0].0, observed[1].0, "candidates carry distinct chunk ordinals");
    assert_ne!(observed[0].1, observed[1].1, "candidates occupy distinct lanes");
    assert_ne!(observed[0].2, observed[1].2, "candidates own distinct fork streams");
}

/// Saturating an attempt's auxiliary envelope never starves lane dispatch:
/// with every permit held elsewhere, the run narrows to width 1 inline and
/// still commits every task exactly once; releasing the permits restores the
/// wide path. The lane gate proves the narrowing structurally: every claim
/// runs on lane 0 while saturated, and both lanes serve work once released.
/// No sleeps: the fallback is a structural property of try-acquire, not
/// timing.
#[test]
fn lane_dispatch_falls_back_to_width_1_when_aux_permits_are_saturated() {
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
    let right =
        builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(Arc::clone(&schema), vec![right_batch])));
    let union = builder.add_union(vec![left, right], Arc::clone(&schema));
    let plan = builder.finish_with_host_sink(union);

    let decision = admit(
        &plan,
        StatisticalAggregateExecutionMode::default(),
        PlanningPolicy::default(),
        rapids_analysis_context(),
    )
    .expect("saturation admission should not error");
    let admission = decision.admission().expect("literal union uses compiled admission");
    let pipeline = admission.pipeline().clone();
    assert!(is_lane_family_pipeline(&pipeline), "literal union dispatches on lanes");

    let runtime = lane_shared_runtime(runtime);
    let test_context = runtime.acquire_context_for_plan(&plan).expect("saturation plan acquires a context");
    let route = test_context.attempt_stream_provider_route().expect("admitted attempt owns a fork route");
    // Hold the whole auxiliary envelope: no lane lease remains for the run.
    let mut held = Vec::new();
    while let Some(lease) =
        test_context.try_acquire_attempt_work_lease().expect("auxiliary try-acquire should not error")
    {
        held.push(lease);
    }
    assert_eq!(held.len(), route.lane_capacity(), "the test holds every auxiliary permit");

    let lanes = Arc::new(Mutex::new(Vec::new()));
    let run = |lanes: &Arc<Mutex<Vec<usize>>>| {
        let metrics = ExecutionMetrics::detached("saturation-fallback");
        let metrics_context = metrics.scope_context(&test_context).expect("metrics scope applies");
        let gpu_inputs = GpuResidentInputMap::new();
        let scheduler = CompiledPipelineScheduler::new(&pipeline).expect("union pipeline is schedulable");
        let mut executor =
            CompiledGpuTaskExecutor::new(&metrics_context, &plan, admission.policy(), &pipeline, &gpu_inputs, &metrics)
                .unwrap();
        let lanes = Arc::clone(lanes);
        let support = LaneDispatchSupport::new(
            Arc::clone(&route),
            metrics_context.clone(),
            plan.clone(),
            admission.policy(),
            pipeline.clone(),
            gpu_inputs.clone(),
            metrics.clone(),
        )
        .with_requested_lanes(2)
        .with_lane_completion_gate(Arc::new(move |lane, _| {
            lanes.lock().expect("saturation lane mutex").push(lane);
        }));
        let state = run_lane_dispatched(&scheduler, &mut executor, None, &support)
            .expect("saturated lane dispatch should complete");
        assert_eq!(state.live_resource_count(), 0, "saturated run releases every resource");
        let batch = executor.take_host_output().expect("saturated sink materializes one batch");
        let mut values = int32_values(&batch, 0);
        values.sort();
        values
    };

    let expected = vec![Some(1), Some(2), Some(3)];
    assert_eq!(run(&lanes), expected, "a saturated attempt still computes every branch once");
    assert!(!lanes.lock().expect("saturation lane mutex").is_empty(), "saturated claims still run");
    assert!(
        lanes.lock().expect("saturation lane mutex").iter().all(|lane| *lane == 0),
        "no claim leaves lane 0 while every permit is held"
    );
    drop(held);
    let lanes = Arc::new(Mutex::new(Vec::new()));
    assert_eq!(run(&lanes), expected, "releasing the envelope restores result parity");
    let observed = lanes.lock().expect("saturation lane mutex");
    assert!(observed.contains(&0) && observed.contains(&1), "released permits serve both lanes: {observed:?}");
}

/// Offset, mixed sort directions, and null keys agree across widths on a
/// single-frame sort input. The frame holds one chunk, so the lane protocol
/// keeps the serial candidate path; this locks the coordinator-inline
/// change to exact parity instead of silent drift.
#[test]
fn lane_dispatch_sort_topk_preserves_offset_mixed_directions_and_nulls() {
    let runtime = cudf::Runtime::new().expect("compiled GPU execution test requires a CUDA runtime");
    let dir = unique_temp_dir("query-engine-lane-sort-offset");
    let keys: Vec<Option<i32>> =
        vec![Some(3), Some(1), Some(2), None, Some(2), Some(3), Some(1), None, Some(0), Some(2)];
    let values: Vec<Option<i64>> = keys.iter().map(|key| key.map(|key| i64::from(key) * 10)).collect();
    let source = sort_lane_nullable_source(&dir.join("input.parquet"), &keys, &values);
    let schema =
        Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, true), Field::new("v", DataType::Int64, true)]));
    let mut builder = QueryPlanBuilder::new();
    let scan = builder.add_source(SourceNode::LocalParquet(source));
    let sort = builder.add_sort(
        scan,
        SortSpec::new(vec![
            SortExpr::new(Expr::column("k"), false, false),
            SortExpr::new(Expr::column("v"), true, false),
        ]),
        Arc::clone(&schema),
    );
    let limit = builder.add_limit(sort, LimitSpec::new(2, 5), Arc::clone(&schema));
    let plan = builder.finish_with_host_sink(limit);

    let decision = admit(
        &plan,
        StatisticalAggregateExecutionMode::default(),
        PlanningPolicy::default(),
        rapids_analysis_context(),
    )
    .expect("offset sort admission should not error");
    let admission = decision.admission().expect("offset sort uses compiled admission");
    let pipeline = admission.pipeline().clone();
    assert!(is_lane_family_pipeline(&pipeline), "single-frame sort dispatches on lanes");

    let runtime = lane_shared_runtime(runtime);
    let test_context = runtime.acquire_context_for_plan(&plan).expect("offset sort acquires a context");
    let route = test_context.attempt_stream_provider_route().expect("admitted sort owns a fork route");
    // A fused top-k over one source is not capped by the DAG leaf count:
    // its input expands into local candidates, so production routing must
    // offer the whole fork pool, not width 1.
    {
        let metrics = ExecutionMetrics::detached("sort-width-request");
        let support = LaneDispatchSupport::new(
            Arc::clone(&route),
            (*test_context).clone(),
            plan.clone(),
            admission.policy(),
            pipeline.clone(),
            GpuResidentInputMap::new(),
            metrics,
        );
        assert_eq!(
            support.requested_lanes,
            route.lane_capacity(),
            "single-source fused top-k requests the full fork pool"
        );
    }
    // A full sort without a fused limit expands nothing: it keeps the leaf
    // cap instead of charging permits for lanes that stay idle.
    {
        let mut full_builder = QueryPlanBuilder::new();
        let full_scan = full_builder.add_source(SourceNode::LocalParquet(sort_lane_nullable_source(
            &dir.join("full.parquet"),
            &[Some(2), Some(1)],
            &[Some(20), Some(10)],
        )));
        let full_sort = full_builder.add_sort(
            full_scan,
            SortSpec::new(vec![SortExpr::new(Expr::column("k"), false, false)]),
            Arc::clone(&schema),
        );
        let full_plan = full_builder.finish_with_host_sink(full_sort);
        let full_pipeline = admit(
            &full_plan,
            StatisticalAggregateExecutionMode::default(),
            PlanningPolicy::default(),
            rapids_analysis_context(),
        )
        .expect("full sort admission should not error")
        .admission()
        .expect("full sort uses compiled admission")
        .pipeline()
        .clone();
        let metrics = ExecutionMetrics::detached("sort-width-request");
        let support = LaneDispatchSupport::new(
            Arc::clone(&route),
            (*test_context).clone(),
            full_plan,
            PlanningPolicy::default(),
            full_pipeline,
            GpuResidentInputMap::new(),
            metrics,
        );
        assert_eq!(support.requested_lanes, 1, "single-source full sort keeps the leaf cap");
    }
    let run = |requested_lanes: Option<usize>| {
        let metrics = ExecutionMetrics::detached("sort-offset-parity");
        let metrics_context = metrics.scope_context(&test_context).expect("metrics scope applies");
        let gpu_inputs = GpuResidentInputMap::new();
        let scheduler = CompiledPipelineScheduler::new(&pipeline).expect("offset sort is schedulable");
        let mut executor =
            CompiledGpuTaskExecutor::new(&metrics_context, &plan, admission.policy(), &pipeline, &gpu_inputs, &metrics)
                .unwrap();
        if let Some(width) = requested_lanes {
            let support = LaneDispatchSupport::new(
                Arc::clone(&route),
                metrics_context.clone(),
                plan.clone(),
                admission.policy(),
                pipeline.clone(),
                gpu_inputs.clone(),
                metrics.clone(),
            )
            .with_requested_lanes(width);
            let state = run_lane_dispatched(&scheduler, &mut executor, None, &support)
                .expect("lane-dispatched offset sort should complete");
            assert_eq!(state.live_resource_count(), 0, "lane run releases every resource");
        } else {
            let state = scheduler.run(&mut executor).expect("serial offset sort should complete");
            assert_eq!(state.live_resource_count(), 0, "serial run releases every resource");
        }
        let batch = executor.take_host_output().expect("offset sort sink materializes one batch");
        int32_values(&batch, 0).into_iter().zip(int64_values(&batch, 1)).collect::<Vec<_>>()
    };

    // Payloads derive from keys, so tied keys carry identical payloads and
    // the unstable tie pick cannot drift the expected rows: k ascending
    // nulls last, v descending, skip 2, fetch 5.
    let mut ordered: Vec<(Option<i32>, Option<i64>)> = keys.into_iter().zip(values).collect();
    ordered.sort_by(|left, right| match (left.0, right.0) {
        (None, None) => std::cmp::Ordering::Equal,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (Some(_), None) => std::cmp::Ordering::Less,
        (Some(left_key), Some(right_key)) => left_key.cmp(&right_key).then_with(|| right.1.cmp(&left.1)),
    });
    let expected: Vec<(Option<i32>, Option<i64>)> = ordered.into_iter().skip(2).take(5).collect();
    assert_eq!(run(None), expected, "serial baseline applies offset after the global order");
    assert_eq!(run(Some(1)), expected, "width 1 preserves offset and mixed directions");
    assert_eq!(run(Some(2)), expected, "width 2 preserves offset and mixed directions");
}
