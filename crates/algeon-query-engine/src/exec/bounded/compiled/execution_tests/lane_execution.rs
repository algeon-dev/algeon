use std::{sync::Condvar, time::Duration};

use ::cudf::testing::device_copy;
use arrow::array::{BooleanArray, Int64Array, StringArray};
use query_planning::{
    optimizer::{NativeOptimizerOptions, optimize_query_plan},
    pipeline::SortStrategy,
    policy::PlanningPolicy,
};

use super::*;
use crate::{
    dimension_pruning::local_dimension_pruning_resolver,
    exec::{
        ExecutionMetrics,
        bounded::compiled::{LaneDispatchSupport, run_resident_lane_borrow},
    },
    rapids_backend::rapids_analysis_context,
};

/// The admitted host pipeline returns every Parquet branch row through the serial scheduler.
#[test]
fn serial_compiled_executes_parquet_union_branches_on_production_path() {
    let runtime = suite_backend::Runtime::new().expect("compiled GPU execution test requires a CUDA runtime");
    let dir = unique_temp_dir("query-engine-lane-parquet-union");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    write_key_value_parquet(&left_path, &[1, 2], &[10, 20]);
    write_key_value_parquet(&right_path, &[3, 4], &[30, 40]);
    let schema =
        Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false), Field::new("v", DataType::Int64, true)]));

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(fixture_local_parquet_source(&left_path, &schema)));
    let left = builder
        .add_projection(
            left,
            vec![
                ProjectionOutput::new(Expr::column("k"), Field::new("k", DataType::Int32, false)),
                ProjectionOutput::new(Expr::column("v"), Field::new("v", DataType::Int64, true)),
            ],
        )
        .unwrap();
    let right = builder.add_source(SourceNode::LocalParquet(fixture_local_parquet_source(&right_path, &schema)));
    let right = builder
        .add_projection(
            right,
            vec![
                ProjectionOutput::new(Expr::column("k"), Field::new("k", DataType::Int32, false)),
                ProjectionOutput::new(Expr::column("v"), Field::new("v", DataType::Int64, true)),
            ],
        )
        .unwrap();
    let union = builder.add_union(vec![left, right], Arc::clone(&schema)).unwrap();
    let limit = builder.add_limit(union, LimitSpec::fetch(10)).unwrap();
    let plan = builder.finish_with_host_sink(limit).unwrap();

    let decision = admit(
        &plan,
        StatisticalAggregateExecutionMode::default(),
        PlanningPolicy::default(),
        rapids_analysis_context(PlanningPolicy::default()),
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
        "the trailing limit stays in the compiled pipeline"
    );

    let runtime = shared_runtime(runtime);
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
    let runtime = suite_backend::Runtime::new().expect("compiled GPU execution test requires a CUDA runtime");
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
    let filter = builder
        .add_filter(scan, Expr::binary(Expr::column("v"), BinaryOp::GtEq, Expr::literal(LiteralValue::Int64(0))))
        .unwrap();
    let projection = builder
        .add_projection(
            filter,
            vec![
                ProjectionOutput::new(Expr::column("k"), Field::new("k", DataType::Int32, false)),
                ProjectionOutput::new(Expr::column("v"), Field::new("measure", DataType::Int64, false)),
            ],
        )
        .unwrap();
    let aggregate = builder
        .add_aggregate(
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
        )
        .unwrap();
    let host_plan = builder.finish_with_host_sink(aggregate).unwrap();
    let plan = host_plan.clone_with_gpu_resident_sink();
    let decision = admit(
        &plan,
        StatisticalAggregateExecutionMode::default(),
        PlanningPolicy::default(),
        rapids_analysis_context(PlanningPolicy::default()),
    )
    .expect("scan/filter/aggregate admission should not error");
    let admission = decision.admission().expect("scan/filter/aggregate uses compiled admission");
    let pipeline = admission.pipeline().clone();

    assert_eq!(pipeline.tasks().iter().filter(|task| task.kind() == PipelineTaskKind::Aggregate).count(), 1);

    let runtime = shared_runtime(runtime);
    {
        let host_context = runtime.acquire_context_for_plan(&host_plan).expect("host aggregate acquires a context");
        let result = execute_query_plan(&host_context, &host_plan).expect("host aggregate executes serially");
        let batch = result.output().as_host_arrow().expect("host aggregate materializes");
        assert_eq!(int64_values(batch, 0), vec![Some(counts.iter().sum())]);
        assert_eq!(int64_values(batch, 1), vec![Some(*minima.iter().min().unwrap())]);
        assert_eq!(int64_values(batch, 2), vec![Some(*maxima.iter().max().unwrap())]);
    }

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
        let mut support = LaneDispatchSupport::new(
            Arc::clone(&route),
            metrics_context,
            plan.clone(),
            admission.policy(),
            pipeline.clone(),
            gpu_inputs,
            metrics,
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
        let (completed, output) =
            run_resident_lane_borrow(&scheduler, None, &support).expect("resident aggregate chunks should complete");
        assert_eq!(completed, pipeline.tasks().len());
        let batch = materialize_gpu_result_to_host(output, &mut QueryMetrics::default())
            .expect("resident aggregate output remains readable after lane release");
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
    let filter = builder
        .add_filter(scan, Expr::binary(Expr::column("v"), BinaryOp::GtEq, Expr::literal(LiteralValue::Int64(0))))
        .unwrap();
    let projection = builder
        .add_projection(
            filter,
            vec![
                ProjectionOutput::new(Expr::column("k"), Field::new("k", DataType::Int32, false)),
                ProjectionOutput::new(Expr::column("v"), Field::new("measure", DataType::Int64, false)),
            ],
        )
        .unwrap();
    let grouped = builder
        .add_aggregate(
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
        )
        .unwrap();
    let grouped_plan = builder.finish_gpu_resident(grouped).unwrap();
    let grouped_decision = admit(
        &grouped_plan,
        StatisticalAggregateExecutionMode::default(),
        PlanningPolicy::default(),
        rapids_analysis_context(PlanningPolicy::default()),
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
    let aggregate = builder
        .add_aggregate(
            scan,
            AggregateSpec::new(
                Vec::new(),
                vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("v"), "rows")],
            ),
            single_schema,
        )
        .unwrap();
    let single_plan = builder.finish_gpu_resident(aggregate).unwrap();
    let single_context = runtime.acquire_context_for_plan(&single_plan).expect("single chunk acquires context");
    let single_result =
        execute_query_plan(&single_context, &single_plan).expect("single chunk aggregate should execute");
    assert_eq!(single_result.metrics().source_chunks_committed, 1);
    let QueryOutput::GpuResident(handle) = single_result.output() else {
        panic!("single chunk aggregate retains its output");
    };
    assert_eq!(
        (handle.frame().placement().stream_debug, handle.frame().placement().memory_resource_debug),
        (handle.runtime().placement().stream_debug, handle.runtime().placement().memory_resource_debug),
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
    let runtime = suite_backend::Runtime::new().expect("compiled GPU execution test requires a CUDA runtime");
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
    let aggregate = builder
        .add_aggregate(
            scan,
            AggregateSpec::new(
                Vec::new(),
                vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("v"), "total")],
            ),
            output_schema,
        )
        .unwrap();
    let plan = builder.finish_gpu_resident(aggregate).unwrap();
    let decision = admit(
        &plan,
        StatisticalAggregateExecutionMode::default(),
        PlanningPolicy::default(),
        rapids_analysis_context(PlanningPolicy::default()),
    )
    .expect("overflow aggregate admission should not error");
    let admission = decision.admission().expect("overflow aggregate uses compiled admission");
    let pipeline = admission.pipeline().clone();

    let runtime = shared_runtime(runtime);
    let test_context = runtime.acquire_context_for_plan(&plan).expect("overflow aggregate acquires context");
    let route = test_context.attempt_stream_provider_route().expect("overflow aggregate has a fork route");
    assert!(route.lane_capacity() >= 4);
    let run = |width: usize| -> std::result::Result<Vec<Option<i64>>, String> {
        let metrics = ExecutionMetrics::detached("aggregate-overflow-lane-parity");
        let metrics_context = metrics.scope_context(&test_context).expect("metrics scope applies");
        let gpu_inputs = GpuResidentInputMap::new();
        let scheduler = CompiledPipelineScheduler::new(&pipeline).expect("overflow pipeline is schedulable");
        let support = LaneDispatchSupport::new(
            Arc::clone(&route),
            metrics_context,
            plan.clone(),
            admission.policy(),
            pipeline.clone(),
            gpu_inputs,
            metrics,
        )
        .with_requested_lanes(width);
        match run_resident_lane_borrow(&scheduler, None, &support) {
            Ok((_, output)) => {
                let batch =
                    materialize_gpu_result_to_host(output, &mut QueryMetrics::default()).expect("overflow sink output");
                Ok(int64_values(&batch, 0))
            },
            Err(error) => Err(error.code().to_string()),
        }
    };
    let expected = run(1);
    assert_eq!(expected, Ok(vec![Some(i64::MIN)]), "single-lane SUM retains the existing wrapping result");
    assert_eq!(run(2), expected, "two lanes preserve the single-lane overflow outcome");
    assert_eq!(run(4), expected, "four lanes preserve the single-lane overflow outcome");
}

/// Larger inner and semi probe batches reuse one build and preserve duplicate rows.
#[test]
fn compiled_join_probe_batches_scale_without_rebuilding() {
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
        let join = builder
            .add_join(
                left,
                right,
                JoinSpec::new(
                    join_type,
                    vec![JoinKey::new(ColumnRef::input(0, "k", "k"), ColumnRef::input(1, "k", "k"))],
                )
                .with_output_projection(projection)
                .with_build_side(build_side),
                output_schema,
            )
            .unwrap();
        let plan = builder.finish_with_host_sink(join).unwrap();
        let runtime = shared_runtime(suite_backend::Runtime::new().expect("join scaling runtime"));
        let context = runtime.acquire_context_for_plan(&plan).expect("join scaling context");
        let result = execute_query_plan(&context, &plan).expect("join scaling executes");
        let batch = result.output().as_host_arrow().expect("join scaling materializes").clone();
        (batch, result.metrics().hash_join_build_count, result.metrics().filtered_join_build_count)
    };
    let (small_batch, builds, _) = run_shape(vec![1, 2], JoinType::Inner);
    assert_eq!(builds, 1, "small probe builds once");
    assert_eq!(int32_int64_pairs_sorted(&small_batch, 0, 1), vec![(Some(1), Some(0)), (Some(2), Some(1))]);
    let (large_batch, builds, _) = run_shape(vec![1, 2, 2, 3, 1, 2, 2, 3], JoinType::Inner);
    assert_eq!(builds, 1, "larger probe batches reuse one build");
    assert_eq!(
        int32_int64_pairs_sorted(&large_batch, 0, 1),
        vec![
            (Some(1), Some(0)),
            (Some(1), Some(4)),
            (Some(2), Some(1)),
            (Some(2), Some(2)),
            (Some(2), Some(5)),
            (Some(2), Some(6)),
        ]
    );
    let (semi_batch, _, builds) = run_shape(vec![1, 2, 2, 3, 1, 2, 2, 3], JoinType::LeftSemi);
    assert_eq!(builds, 1, "semi probe batches reuse one filtered build");
    assert_eq!(
        int32_int64_pairs_sorted(&semi_batch, 0, 1),
        vec![
            (Some(1), Some(0)),
            (Some(1), Some(4)),
            (Some(2), Some(1)),
            (Some(2), Some(2)),
            (Some(2), Some(5)),
            (Some(2), Some(6)),
        ]
    );
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

/// A chained semi join feeds four hash-probe chunks through the serial DAG coordinator.
/// The join owner borrows auxiliary lanes and reuses one build for those chunks.
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
    let join1 = builder
        .add_join(
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
        )
        .unwrap();
    let b2 = builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(
        Arc::clone(&key_schema),
        vec![
            RecordBatch::try_new(Arc::clone(&key_schema), vec![Arc::new(Int32Array::from(vec![2])) as ArrayRef])
                .expect("hash build batch"),
        ],
    )));
    let join2 = builder
        .add_join(
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
        )
        .unwrap();
    let plan = builder.finish_with_host_sink(join2).unwrap();
    let decision = admit(
        &plan,
        StatisticalAggregateExecutionMode::default(),
        PlanningPolicy::default(),
        rapids_analysis_context(PlanningPolicy::default()),
    )
    .expect("chained join admission should not error");
    let admission = decision.admission().expect("chained joins use compiled admission");
    let pipeline = admission.pipeline().clone();
    assert_eq!(
        pipeline.tasks().iter().filter(|task| task.kind() == PipelineTaskKind::JoinBuild).count(),
        2,
        "two joins build separately"
    );
    let runtime = shared_runtime(suite_backend::Runtime::new().expect("chained join runtime"));
    let test_context = runtime.acquire_context_for_plan(&plan).expect("chained join context");
    let run = || {
        let metrics = ExecutionMetrics::detached("chained-hash-parallel");
        let metrics_context = metrics.scope_context(&test_context).expect("metrics scope applies");
        let gpu_inputs = GpuResidentInputMap::new();
        let scheduler = CompiledPipelineScheduler::new(&pipeline).expect("chained pipeline is schedulable");
        let mut executor =
            CompiledGpuTaskExecutor::new(&metrics_context, &plan, admission.policy(), &pipeline, &gpu_inputs, &metrics)
                .unwrap();
        let state = scheduler.run(&mut executor).expect("compiled join should execute");
        assert_eq!(state.live_resource_count(), 0);
        let batch = executor.take_host_output().expect("compiled join materializes");
        (batch, metrics.snapshot())
    };
    let wide_len = "x".repeat(700_000).len();
    let expected = vec![(Some(2), wide_len), (Some(2), wide_len), (Some(2), wide_len), (Some(2), wide_len)];
    let (batch, snapshot) = run();
    let label = "serial";
    assert_eq!(int32_strlens_sorted(&batch), expected, "{label} joins every chained probe row once");
    assert_eq!(snapshot.hash_join_build_count, 1, "{label} builds the hash table once");
    assert_eq!(snapshot.filtered_join_build_count, 1, "{label} builds the filtered table once");
    assert_eq!(
        snapshot.source_chunks_committed, 4,
        "{label} reads four probe files as four chunks feeding four probe batches"
    );
    let probe_snapshot = snapshot;
    let parallel = probe_snapshot
        .join_strategy_decisions
        .iter()
        .find(|decision| decision.facts.get("parallel_probe").map(String::as_str) == Some("hash"))
        .expect("serial coordinator records the hash parallel-probe decision");
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

/// A chained semi join feeds four probe chunks through one partitioned build.
/// The join owner retains its chunk lane dispatch under the serial DAG coordinator.
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
    let join1 = builder
        .add_join(
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
        )
        .unwrap();
    let mut build_file = enrich_fixture_local_parquet_file(LocalParquetFile::new(&build_path));
    build_file.row_group_rows = Some(vec![100_000_000]);
    let b2 = builder.add_source(SourceNode::LocalParquet(NativeLocalParquetSource::from_files(
        vec![build_file],
        Arc::clone(&build_schema),
        vec!["k".to_string(), "w".to_string()],
    )));
    let join2 = builder
        .add_join(
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
        )
        .unwrap();
    let plan = builder.finish_with_host_sink(join2).unwrap();
    let decision = admit(
        &plan,
        StatisticalAggregateExecutionMode::default(),
        PlanningPolicy::default(),
        rapids_analysis_context(PlanningPolicy::default()),
    )
    .expect("partitioned chain admission should not error");
    let admission = decision.admission().expect("partitioned chain uses compiled admission");
    let pipeline = admission.pipeline().clone();
    let runtime = shared_runtime(suite_backend::Runtime::new().expect("partitioned chain runtime"));
    let test_context = runtime.acquire_context_for_plan(&plan).expect("partitioned chain context");
    let run = || {
        let metrics = ExecutionMetrics::detached("chained-partitioned-parallel");
        let metrics_context = metrics.scope_context(&test_context).expect("metrics scope applies");
        let gpu_inputs = GpuResidentInputMap::new();
        let scheduler = CompiledPipelineScheduler::new(&pipeline).expect("partitioned pipeline is schedulable");
        let mut executor =
            CompiledGpuTaskExecutor::new(&metrics_context, &plan, admission.policy(), &pipeline, &gpu_inputs, &metrics)
                .unwrap();
        let state = scheduler.run(&mut executor).expect("compiled join should execute");
        assert_eq!(state.live_resource_count(), 0);
        let batch = executor.take_host_output().expect("compiled join materializes");
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
    let (batch, snapshot) = run();
    let label = "serial";
    assert_eq!(int32_strlen_int64_triples_sorted(&batch), expected, "{label} joins every partitioned probe row once");
    assert_eq!(snapshot.hash_join_build_count, 0, "{label} builds no hash table");
    assert_eq!(
        snapshot.source_chunks_committed, 5,
        "{label} reads four probe files plus one build file as five chunks"
    );
    let probe_snapshot = snapshot;
    assert!(
        probe_snapshot.join_strategy_decisions.iter().any(|decision| {
            decision.facts.get("join_strategy").map(String::as_str) == Some("partitioned_hash_equi")
        }),
        "second join plans a split partitioned build"
    );
    let parallel = probe_snapshot
        .join_strategy_decisions
        .iter()
        .find(|decision| decision.facts.get("parallel_probe").map(String::as_str) == Some("partitioned"))
        .expect("serial coordinator records the partitioned parallel-probe decision");
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

/// Host and resident sinks preserve the same null-key, duplicate, and residual join results.
/// The expected rows also cover mark membership and full-outer complements.
#[test]
fn host_and_resident_sinks_preserve_join_kinds_with_nulls_duplicates_and_residuals() {
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

    fn execute_pair(plan: &QueryPlan) -> (RecordBatch, RecordBatch) {
        let runtime = shared_runtime(suite_backend::Runtime::new().expect("parity runtime"));
        let host = {
            let context = runtime.acquire_context_for_plan(plan).expect("host parity context");
            let result = execute_query_plan(&context, plan).expect("host join executes");
            result.output().as_host_arrow().expect("host parity output").clone()
        };
        let resident_plan = plan.clone_with_gpu_resident_sink();
        let context = runtime.acquire_context_for_plan(&resident_plan).expect("resident parity context");
        let result = execute_query_plan(&context, &resident_plan).expect("resident join executes");
        let output = result.output().as_gpu_resident().expect("resident parity output").clone();
        drop(result);
        let resident =
            materialize_gpu_result_to_host(output, &mut QueryMetrics::default()).expect("resident join materializes");
        (host, resident)
    }

    fn admitted_pipeline(label: &str, plan: &QueryPlan) -> CompiledPipeline {
        admit(
            plan,
            StatisticalAggregateExecutionMode::default(),
            PlanningPolicy::default(),
            rapids_analysis_context(PlanningPolicy::default()),
        )
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
        ("anti", JoinType::LEFT_ANTI, JoinSide::Right, JoinNullEquality::NullsDoNotMatch, false),
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
            JoinType::LeftSemi | JoinType::LeftAnti { .. } => (
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
        let mut spec = JoinSpec::new(
            join_type.clone(),
            vec![JoinKey::new(ColumnRef::input(0, "k", "k"), ColumnRef::input(1, "k", "k"))],
        )
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
        let join = builder.add_join(left, right, spec, output_schema).unwrap();
        let plan = builder.finish_with_host_sink(join).unwrap();
        let _pipeline = admitted_pipeline(label, &plan);
        let (serial, resident) = execute_pair(&plan);

        match join_type {
            JoinType::LeftSemi | JoinType::LeftAnti { .. } => {
                let expected = int32_int64_pairs_sorted(&serial, 0, 1);
                assert_eq!(
                    int32_int64_pairs_sorted(&resident, 0, 1),
                    expected,
                    "{label}: resident sink preserves rows"
                );

                let oracle = match join_type {
                    JoinType::LeftSemi => vec![(Some(2), Some(30)), (Some(2), Some(40))],
                    _ => vec![(None, Some(20)), (Some(1), Some(10))],
                };
                assert_eq!(expected, oracle, "{label}: serial matches the null-key oracle");
            },
            _ => {
                let expected = int32_int64_int64_triples_sorted(&serial, 0, 1, 2);
                assert_eq!(
                    int32_int64_int64_triples_sorted(&resident, 0, 1, 2),
                    expected,
                    "{label}: resident sink preserves rows"
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
    let mark = builder
        .add_join(
            left,
            right,
            JoinSpec::new(
                JoinType::LeftMark { mark_column: "mark".to_string() },
                vec![JoinKey::new(ColumnRef::input(0, "k", "k"), ColumnRef::input(1, "k", "k"))],
            )
            .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "k", "k"), "k")])
            .with_null_equality(JoinNullEquality::NullsDoNotMatch)
            .with_build_side(JoinSide::Right),
            Arc::clone(&mark_schema),
        )
        .unwrap();
    let plan = builder.finish_with_host_sink(mark).unwrap();
    let _pipeline = admitted_pipeline("mark", &plan);
    let (serial, resident) = execute_pair(&plan);
    let oracle = vec![(None, Some(false)), (Some(1), Some(false)), (Some(2), Some(true)), (Some(2), Some(true))];
    assert_eq!(int32_bool_pairs_sorted(&serial), oracle, "mark: serial matches the membership oracle");
    assert_eq!(int32_bool_pairs_sorted(&resident), oracle, "mark: resident sink preserves membership");
}

/// The compiled control dependency completes the build before the probe source.
/// The source applies its dynamic filter and reads one matching row group.
#[test]
fn serial_compiled_join_preserves_dynamic_filter_pruning() {
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
    let join = builder
        .add_join(
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
        )
        .unwrap();
    let raw = builder.finish_with_host_sink(join).unwrap();
    let optimized = optimize_query_plan(
        raw,
        &NativeOptimizerOptions::default(),
        rapids_analysis_context(PlanningPolicy::default()),
        local_dimension_pruning_resolver(),
    )
    .unwrap();
    assert!(
        !optimized.plan.dynamic_filter_routes().is_empty(),
        "optimizer discovers a build-to-probe dynamic-filter route"
    );
    let plan = optimized.plan;
    let decision = admit(
        &plan,
        StatisticalAggregateExecutionMode::default(),
        PlanningPolicy::default(),
        rapids_analysis_context(PlanningPolicy::default()),
    )
    .expect("pruning admission should not error");
    let admission = decision.admission().expect("pruning shape uses compiled admission");
    let pipeline = admission.pipeline().clone();
    let runtime = shared_runtime(suite_backend::Runtime::new().expect("pruning runtime"));
    let test_context = runtime.acquire_context_for_plan(&plan).expect("pruning context");
    let run = || {
        let metrics = ExecutionMetrics::detached("join-lane-pruning");
        let metrics_context = metrics.scope_context(&test_context).expect("metrics scope applies");
        let gpu_inputs = GpuResidentInputMap::new();
        let scheduler = CompiledPipelineScheduler::new(&pipeline).expect("pruning pipeline is schedulable");
        let mut executor =
            CompiledGpuTaskExecutor::new(&metrics_context, &plan, admission.policy(), &pipeline, &gpu_inputs, &metrics)
                .unwrap();
        let state = scheduler.run(&mut executor).expect("compiled join should execute");
        assert_eq!(state.live_resource_count(), 0);
        let batch = executor.take_host_output().expect("compiled join materializes");
        (batch, metrics.snapshot())
    };
    let expected = vec![(Some(2), Some(2)), (Some(2), Some(3))];
    let (batch, snapshot) = run();
    let label = "serial";
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

/// A resident join with an aggregate retains coordinator-owned join state.
/// Its chunk lanes stay alive through output adoption onto the root runtime.
#[test]
fn resident_sink_join_with_aggregate_routes_through_resident_lanes() {
    let runtime = suite_backend::Runtime::new().expect("compiled GPU execution test requires a CUDA runtime");
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
    let join = builder
        .add_join(
            build,
            probe,
            JoinSpec::new(
                JoinType::Inner,
                vec![JoinKey::new(ColumnRef::input(0, "k", "k"), ColumnRef::input(1, "k", "k"))],
            )
            .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "k", "k"), "k")])
            .with_build_side(JoinSide::Left),
            Arc::clone(&schema),
        )
        .unwrap();
    let aggregate = builder
        .add_aggregate(
            join,
            AggregateSpec::new(
                Vec::new(),
                vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("k"), "rows")],
            ),
            Arc::clone(&count_schema),
        )
        .unwrap();
    let plan = builder.finish_gpu_resident(aggregate).unwrap();
    let decision = admit(
        &plan,
        StatisticalAggregateExecutionMode::default(),
        PlanningPolicy::default(),
        rapids_analysis_context(PlanningPolicy::default()),
    )
    .expect("resident admission should not error");
    assert!(decision.admission().is_some(), "resident join uses compiled admission");
    let runtime = shared_runtime(runtime);
    let test_context = runtime.acquire_context_for_plan(&plan).expect("resident join acquires a context");
    let result = execute_query_plan(&test_context, &plan).expect("resident join lanes should execute");
    let QueryOutput::GpuResident(handle) = result.output() else {
        panic!("resident join retains its output");
    };
    assert_eq!(
        (handle.frame().placement().stream_debug, handle.frame().placement().memory_resource_debug),
        (handle.runtime().placement().stream_debug, handle.runtime().placement().memory_resource_debug),
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
    let runtime = suite_backend::Runtime::new().expect("compiled GPU execution test requires a CUDA runtime");
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
    let aggregate = builder
        .add_aggregate(
            scan,
            AggregateSpec::new(
                Vec::new(),
                vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("v"), "total")],
            ),
            Arc::new(Schema::new(vec![Field::new("total", DataType::Int64, true)])),
        )
        .unwrap();
    let plan = builder.finish_gpu_resident(aggregate).unwrap();

    let runtime = shared_runtime(runtime);
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

/// Resident top-k candidate lanes preserve the host serial result, including null ordering.
#[test]
fn lane_dispatch_sort_topk_candidates_overlap_lanes_and_match_serial() {
    let runtime = suite_backend::Runtime::new().expect("compiled GPU execution test requires a CUDA runtime");
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
    let union = builder.add_union(vec![left, right], Arc::clone(&schema)).unwrap();
    let sort = builder.add_sort(union, SortSpec::new(vec![SortExpr::new(Expr::column("k"), true, true)])).unwrap();
    let limit = builder.add_limit(sort, LimitSpec::new(0, 6)).unwrap();
    let plan = builder.finish_with_host_sink(limit).unwrap();

    let decision = admit(
        &plan,
        StatisticalAggregateExecutionMode::default(),
        PlanningPolicy::default(),
        rapids_analysis_context(PlanningPolicy::default()),
    )
    .expect("sort/top-k admission should not error");
    let admission = decision.admission().expect("union sort/top-k uses compiled admission");
    let pipeline = admission.pipeline().clone();

    let sort_task = pipeline
        .tasks()
        .iter()
        .find(|task| task.kind() == PipelineTaskKind::FinalSort)
        .expect("union sort/top-k compiles a final sort task");
    assert!(
        matches!(sort_task.sort_strategy(), Some(SortStrategy::FusedTopK { .. })),
        "sort/limit fuses into a stream-capable top-k"
    );

    let runtime = shared_runtime(runtime);
    let test_context = runtime.acquire_context_for_plan(&plan).expect("sort/top-k plan acquires a context");
    let route = test_context.attempt_stream_provider_route().expect("admitted sort owns a fork route");
    let arrivals = Arc::new((Mutex::new(Vec::new()), Condvar::new()));
    let run = |requested_lanes: Option<usize>, coordinate: bool| {
        let metrics = ExecutionMetrics::detached("sort-lane-parity");
        let metrics_context = metrics.scope_context(&test_context).expect("metrics scope applies");
        let gpu_inputs = GpuResidentInputMap::new();
        let scheduler = CompiledPipelineScheduler::new(&pipeline).expect("sort pipeline is schedulable");
        let batch = if let Some(width) = requested_lanes {
            let resident_plan = plan.clone_with_gpu_resident_sink();
            let analysis = resident_plan.capability_analysis(rapids_analysis_context(admission.policy()));
            let resident_pipeline = compile_pipeline(&resident_plan, &analysis).expect("resident top-k compiles");
            let resident_scheduler =
                CompiledPipelineScheduler::new(&resident_pipeline).expect("resident top-k is schedulable");
            let mut support = LaneDispatchSupport::new(
                Arc::clone(&route),
                metrics_context,
                resident_plan,
                admission.policy(),
                resident_pipeline.clone(),
                gpu_inputs,
                metrics,
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
            let (_, output) = run_resident_lane_borrow(&resident_scheduler, None, &support)
                .expect("resident sort candidates should complete");
            materialize_gpu_result_to_host(output, &mut QueryMetrics::default()).expect("resident sort materializes")
        } else {
            let mut executor = CompiledGpuTaskExecutor::new(
                &metrics_context,
                &plan,
                admission.policy(),
                &pipeline,
                &gpu_inputs,
                &metrics,
            )
            .unwrap();
            let state = scheduler.run(&mut executor).expect("serial sort should complete");
            assert_eq!(state.live_resource_count(), 0);
            executor.take_host_output().expect("host sort materializes")
        };
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

/// Resident top-k borrowing preserves the host serial offset and mixed-direction ordering.
#[test]
fn lane_dispatch_sort_topk_preserves_offset_mixed_directions_and_nulls() {
    let runtime = suite_backend::Runtime::new().expect("compiled GPU execution test requires a CUDA runtime");
    let dir = unique_temp_dir("query-engine-lane-sort-offset");
    let keys: Vec<Option<i32>> =
        vec![Some(3), Some(1), Some(2), None, Some(2), Some(3), Some(1), None, Some(0), Some(2)];
    let values: Vec<Option<i64>> = keys.iter().map(|key| key.map(|key| i64::from(key) * 10)).collect();
    let source = sort_lane_nullable_source(&dir.join("input.parquet"), &keys, &values);
    let mut builder = QueryPlanBuilder::new();
    let scan = builder.add_source(SourceNode::LocalParquet(source));
    let sort = builder
        .add_sort(
            scan,
            SortSpec::new(vec![
                SortExpr::new(Expr::column("k"), false, false),
                SortExpr::new(Expr::column("v"), true, false),
            ]),
        )
        .unwrap();
    let limit = builder.add_limit(sort, LimitSpec::new(2, 5)).unwrap();
    let plan = builder.finish_with_host_sink(limit).unwrap();

    let decision = admit(
        &plan,
        StatisticalAggregateExecutionMode::default(),
        PlanningPolicy::default(),
        rapids_analysis_context(PlanningPolicy::default()),
    )
    .expect("offset sort admission should not error");
    let admission = decision.admission().expect("offset sort uses compiled admission");
    let pipeline = admission.pipeline().clone();

    let runtime = shared_runtime(runtime);
    let test_context = runtime.acquire_context_for_plan(&plan).expect("offset sort acquires a context");
    let route = test_context.attempt_stream_provider_route().expect("admitted sort owns a fork route");
    let run = |requested_lanes: Option<usize>| {
        let metrics = ExecutionMetrics::detached("sort-offset-parity");
        let metrics_context = metrics.scope_context(&test_context).expect("metrics scope applies");
        let gpu_inputs = GpuResidentInputMap::new();
        let scheduler = CompiledPipelineScheduler::new(&pipeline).expect("offset sort is schedulable");
        let batch = if let Some(width) = requested_lanes {
            let resident_plan = plan.clone_with_gpu_resident_sink();
            let analysis = resident_plan.capability_analysis(rapids_analysis_context(admission.policy()));
            let resident_pipeline = compile_pipeline(&resident_plan, &analysis).expect("resident top-k compiles");
            let resident_scheduler =
                CompiledPipelineScheduler::new(&resident_pipeline).expect("resident top-k is schedulable");
            let support = LaneDispatchSupport::new(
                Arc::clone(&route),
                metrics_context,
                resident_plan,
                admission.policy(),
                resident_pipeline.clone(),
                gpu_inputs,
                metrics,
            )
            .with_requested_lanes(width);
            let (_, output) = run_resident_lane_borrow(&resident_scheduler, None, &support)
                .expect("resident sort candidates should complete");
            materialize_gpu_result_to_host(output, &mut QueryMetrics::default()).expect("resident sort materializes")
        } else {
            let mut executor = CompiledGpuTaskExecutor::new(
                &metrics_context,
                &plan,
                admission.policy(),
                &pipeline,
                &gpu_inputs,
                &metrics,
            )
            .unwrap();
            let state = scheduler.run(&mut executor).expect("serial sort should complete");
            assert_eq!(state.live_resource_count(), 0);
            executor.take_host_output().expect("host sort materializes")
        };
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
