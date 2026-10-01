use std::sync::Arc;

use query_engine::{
    error::Result as QueryEngineResult,
    exec::{
        CompiledGpuResource, CompiledGpuTaskExecutor, CompiledPipelineScheduler, CompiledPipelineState,
        CompiledTaskExecutor, ExecutionMetrics,
    },
    expr::ColumnRef,
    pipeline::{PipelineTask, PipelineTaskKind, ResidentResourceId, compile_pipeline},
    plan::NodeId,
    policy::PlanningPolicy,
    rapids_backend::rapids_analysis_context,
};

use super::*;

#[test]
fn bounded_native_query_executes_global_final_merge_with_consistent_state_schema() {
    let dir = unique_temp_dir("query-engine-bounded-global-final-merge-retry");
    let path = dir.join("input.parquet");
    let input_schema = write_final_merge_retry_input_parquet(&path, 1024);
    let repeated_sums = 256;
    let mut fields = (0..repeated_sums)
        .map(|index| Field::new(format!("sum_m00_{index}"), DataType::Int64, true))
        .collect::<Vec<_>>();
    fields.extend([
        Field::new("min_m00", DataType::Int64, true),
        Field::new("max_m00", DataType::Int64, true),
        Field::new("count_m00", DataType::Int64, false),
    ]);
    let aggregated_schema = Arc::new(Schema::new(fields));

    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["m00".to_string()],
    );
    let mut aggr_exprs = (0..repeated_sums)
        .map(|index| AggregateExpr::new(AggregateFunction::Sum, Expr::column("m00"), format!("sum_m00_{index}")))
        .collect::<Vec<_>>();
    aggr_exprs.extend([
        AggregateExpr::new(AggregateFunction::Min, Expr::column("m00"), "min_m00"),
        AggregateExpr::new(AggregateFunction::Max, Expr::column("m00"), "max_m00"),
        AggregateExpr::new(AggregateFunction::Count, Expr::column("m00"), "count_m00"),
    ]);
    let aggregate = AggregateSpec::new(Vec::new(), aggr_exprs);

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let aggregate = builder.add_aggregate(source, aggregate, Arc::clone(&aggregated_schema));
    let plan = builder.finish_with_host_sink(aggregate);

    let context = backend_context_with_memory_control_policy(MemoryControlPolicy {
        device_budget_bytes: 4_000_000,
        max_source_chunk_bytes: 1024 * 1024,
        max_row_groups_per_chunk: 1,
        max_retry_attempts: 3,
        min_retry_chunk_row_groups: 1,
    });

    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.source_chunks_planned, 3);
    assert_eq!(metrics.source_chunks_committed, 3);
    assert_eq!(metrics.final_merge_attempts, 1);
    assert_eq!(metrics.final_merge_levels, 1);
    assert_eq!(metrics.retry_count, 0);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 1);
    assert_eq!(batch.num_columns(), repeated_sums + 3);
    assert!(
        batch.schema().fields().iter().all(|field| !field.name().starts_with("__algeon_native_global_valid_count_"))
    );
    for index in 0..repeated_sums {
        assert_eq!(int64_array(batch, index).value(0), 3072);
    }
    assert_eq!(int64_array(batch, repeated_sums).value(0), 1);
    assert_eq!(int64_array(batch, repeated_sums + 1).value(0), 1);
    assert_eq!(int64_array(batch, repeated_sums + 2).value(0), 3072);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_context_executes_wide_global_final_merge_without_partitioned_path() {
    let dir = unique_temp_dir("query-engine-raw-bounded-final-merge-retry");
    let path = dir.join("input.parquet");
    let input_schema = write_final_merge_retry_input_parquet(&path, 1024);
    let repeated_sums = 256;
    let aggregated_schema = Arc::new(Schema::new(
        (0..repeated_sums)
            .map(|index| Field::new(format!("sum_m00_{index}"), DataType::Int64, true))
            .collect::<Vec<_>>(),
    ));

    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["m00".to_string()],
    );
    let aggregate = AggregateSpec::new(
        Vec::new(),
        (0..repeated_sums)
            .map(|index| AggregateExpr::new(AggregateFunction::Sum, Expr::column("m00"), format!("sum_m00_{index}")))
            .collect::<Vec<_>>(),
    );

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let aggregate = builder.add_aggregate(source, aggregate, Arc::clone(&aggregated_schema));
    let plan = builder.finish_with_host_sink(aggregate);

    let context = backend_context_with_memory_control_policy(MemoryControlPolicy {
        device_budget_bytes: NATIVE_EXECUTION_TEST_QUERY_LIMIT_BYTES,
        max_source_chunk_bytes: 1024 * 1024,
        max_row_groups_per_chunk: 1,
        max_retry_attempts: 3,
        min_retry_chunk_row_groups: 1,
    });

    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.source_chunks_planned, 3);
    assert_eq!(metrics.source_chunks_committed, 3);
    assert_eq!(metrics.final_merge_attempts, 1);
    assert_eq!(metrics.final_merge_levels, 1);
    assert_eq!(metrics.retry_count, 0);
    assert_eq!(metrics.retry_resource_exhausted_count, 0);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 1);
    assert_eq!(batch.num_columns(), repeated_sums);
    for index in 0..repeated_sums {
        assert_eq!(int64_array(batch, index).value(0), 3072);
    }

    fs::remove_dir_all(dir).unwrap();
}

fn incremental_partitioned_merge_builder(
    dir: &Path,
    row_groups: usize,
    rows_per_row_group: usize,
) -> (QueryPlanBuilder, NodeId, Arc<Schema>) {
    let path = dir.join(format!("input-{row_groups}.parquet"));
    let input_schema = write_unique_key_row_grouped_parquet(&path, row_groups, rows_per_row_group);
    let aggregated_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int64, false),
        Field::new("sum_value", DataType::Int64, true),
        Field::new("count_value", DataType::Int64, false),
        Field::new("avg_value", DataType::Float64, true),
    ]));
    let dimension_path = dir.join("dimension.parquet");
    let dimension_schema = write_single_row_dimension_parquet(&dimension_path);
    let source = fixture_local_parquet_source::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["key".to_string(), "value".to_string(), "d".to_string()],
    );
    let dimension = fixture_local_parquet_source::from_files(
        vec![LocalParquetFile::new(&dimension_path).with_file_size_bytes(fs::metadata(&dimension_path).unwrap().len())],
        Arc::clone(&dimension_schema),
        vec!["d".to_string()],
    );
    let join_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int64, false),
        Field::new("value", DataType::Int64, false),
    ]));
    // COUNT and AVG carry partial state whose column shape differs from the
    // merged state, so merging raw slices with merged states in one set would
    // fail on schema mismatch.
    let aggregate = AggregateSpec::new(
        vec![Expr::column("key")],
        vec![
            AggregateExpr::new(AggregateFunction::Sum, Expr::column("value"), "sum_value"),
            AggregateExpr::new(AggregateFunction::Count, Expr::column("value"), "count_value"),
            AggregateExpr::new(AggregateFunction::Avg, Expr::column("value"), "avg_value"),
        ],
    );
    // The aggregate consumes the streamed probe output of a compiled join, the
    // shape whose partials arrive one chunk at a time.
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let dimension = builder.add_source(SourceNode::LocalParquet(dimension));
    let join = builder.add_join(
        source,
        dimension,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "d", "d"), ColumnRef::input(1, "d", "d"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "key", "key"), "key"),
            JoinOutputColumn::left(ColumnRef::input(0, "value", "value"), "value"),
        ])
        .with_build_side(JoinSide::Right),
        Arc::clone(&join_schema),
    );
    let aggregate = builder.add_aggregate(join, aggregate, Arc::clone(&aggregated_schema));
    (builder, aggregate, aggregated_schema)
}

fn incremental_partitioned_merge_plans(
    dir: &Path,
    row_groups: usize,
    rows_per_row_group: usize,
) -> (QueryPlan, QueryPlan) {
    let (mut builder, aggregate, aggregated_schema) =
        incremental_partitioned_merge_builder(dir, row_groups, rows_per_row_group);
    let plan = builder.clone().finish_with_host_sink(aggregate);
    let sort = builder.add_sort(
        aggregate,
        SortSpec::new(vec![SortExpr::new(Expr::column("key"), false, false)]),
        Arc::clone(&aggregated_schema),
    );
    let limit = builder.add_limit(sort, LimitSpec::new(5, Some(10)), aggregated_schema);
    (plan, builder.finish_with_host_sink(limit))
}

#[test]
fn bounded_native_query_merges_high_cardinality_partials_incrementally_under_the_query_cap() {
    let dir = unique_temp_dir("query-engine-bounded-incremental-partitioned-merge");
    let row_groups = 32;
    let rows_per_row_group = 131_072;
    let (plan, _) = incremental_partitioned_merge_plans(&dir, row_groups, rows_per_row_group);

    // Every row group keeps all of its rows as partial state (about 160 MiB in
    // total across SUM, COUNT, and AVG state). Retaining every partial beside
    // its hash-partition slices and then the merged partitions beside their
    // concatenation needs three times the merged state; folding partitions
    // incrementally keeps the retained set at the merged state plus its final
    // concatenation.
    let context = backend_context_with_memory_control_policy(MemoryControlPolicy {
        device_budget_bytes: 400 * 1024 * 1024,
        max_source_chunk_bytes: 8 * 1024 * 1024,
        max_row_groups_per_chunk: 1,
        max_retry_attempts: 3,
        min_retry_chunk_row_groups: 1,
    });
    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.compiled_pipeline_attempts, 1);
    // The probe side streams in source batches, so partial count follows the
    // batch plan rather than the row-group count.
    assert!(metrics.partial_aggregate_count >= 2, "{metrics:?}");
    let final_merge = metrics
        .aggregate_strategy_decisions
        .iter()
        .find(|decision| decision.facts.contains_key("partitioned_merge_partition_count"))
        .expect("partitioned final merge strategy row");
    assert_eq!(final_merge.facts.get("partitioned_merge_incremental").map(String::as_str), Some("true"));
    assert!(
        final_merge.facts.get("partitioned_merge_incremental_flushes").and_then(|value| value.parse::<u64>().ok())
            > Some(1),
        "{:?}",
        final_merge.facts
    );

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), row_groups * rows_per_row_group);
    let keys = int64_array(batch, 0);
    let sums = int64_array(batch, 1);
    let counts = int64_array(batch, 2);
    let averages = float64_array(batch, 3);
    assert!((0..batch.num_rows()).all(|row| {
        sums.value(row) == keys.value(row) % 7
            && counts.value(row) == 1
            && averages.value(row) == (keys.value(row) % 7) as f64
    }));

    // The larger state selects more hash partitions without changing policy.
    // Its 512 MiB final output cannot coexist with a full concatenation under
    // this cap, while each finalized partition fits beside the retained state.
    let (eager_plan, streamed_plan) = incremental_partitioned_merge_plans(&dir, 128, rows_per_row_group);
    let cap = 960 * 1024 * 1024;
    let streamed_context = backend_context_with_memory_control_policy(MemoryControlPolicy {
        device_budget_bytes: cap,
        max_source_chunk_bytes: 8 * 1024 * 1024,
        max_row_groups_per_chunk: 1,
        max_retry_attempts: 3,
        min_retry_chunk_row_groups: 1,
    });
    let admitted = streamed_context.admit_plan(&streamed_plan).unwrap();
    let admitted_plan =
        admit_query_plan(&streamed_plan, admitted.statistical_aggregate_execution_mode(), admitted.planning_policy())
            .unwrap();
    let outcome = execute_query_plan_direct(
        admitted.context.context(),
        QueryExecutionRequest::admitted(&admitted_plan).with_gpu_inputs(&GpuResidentInputMap::new()),
    );
    let (output, metrics, _) = outcome.into_parts();
    let output = output.unwrap();
    let usage = admitted.attempt_allocation_domain_usage().unwrap().unwrap();
    assert!(usage.peak_bytes <= cap, "{usage:?}");
    assert_eq!(usage.current_bytes, 0, "all partition state must be released after host output: {usage:?}");
    assert_eq!(usage.current_allocations, 0);
    assert!(
        metrics.aggregate_strategy_decisions.iter().any(|decision| decision
            .facts
            .get("partitioned_merge_streamed_finalization")
            .map(String::as_str)
            == Some("true"))
    );
    let QueryOutput::HostArrow(batch) = output else { panic!("host top-k output") };
    assert_eq!(batch.num_rows(), 10);
    for row in 0..10 {
        let key = (row + 5) as i64;
        assert_eq!(int64_array(&batch, 0).value(row), key);
        assert_eq!(int64_array(&batch, 1).value(row), key % 7);
        assert_eq!(int64_array(&batch, 2).value(row), 1);
        assert_eq!(float64_array(&batch, 3).value(row), (key % 7) as f64);
    }
    admitted.finish();
    drop(admitted);

    let admitted = streamed_context.admit_plan(&eager_plan).unwrap();
    let admitted_plan =
        admit_query_plan(&eager_plan, admitted.statistical_aggregate_execution_mode(), admitted.planning_policy())
            .unwrap();
    let outcome = execute_query_plan_direct(
        admitted.context.context(),
        QueryExecutionRequest::admitted(&admitted_plan).with_gpu_inputs(&GpuResidentInputMap::new()),
    );
    let (output, _, _) = outcome.into_parts();
    let error = output.err().expect("full final-output concatenation exceeds the same cap");
    assert_eq!(error.code(), QueryErrorCode::QueryMemoryLimitExceeded);
    assert_eq!(error.operation(), "query_engine.final_merge.partitioned_output_concat");
    assert_attempt_domain_released(admitted.context.context());
    admitted.finish();

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn compiled_final_aggregate_fanout_keeps_independent_consumers_and_releases_shared_state() {
    let dir = unique_temp_dir("query-engine-final-aggregate-fanout");
    let (mut builder, aggregate, schema) = incremental_partitioned_merge_builder(&dir, 128, 131_072);
    let projected = builder.add_projection(
        aggregate,
        schema
            .fields()
            .iter()
            .map(|field| ProjectionOutput::new(Expr::column(field.name()), field.as_ref().clone()))
            .collect(),
    );
    let filtered = builder.add_filter(
        projected,
        Expr::binary(Expr::column("count_value"), BinaryOp::Gt, Expr::literal(LiteralValue::Int64(0))),
        Arc::clone(&schema),
    );
    let left_sort = builder.add_sort(
        filtered,
        SortSpec::new(vec![SortExpr::new(Expr::column("key"), false, false)]),
        Arc::clone(&schema),
    );
    let left = builder.add_limit(left_sort, LimitSpec::new(5, Some(10)), Arc::clone(&schema));
    let right_sort = builder.add_sort(
        projected,
        SortSpec::new(vec![SortExpr::new(Expr::column("key"), false, false)]),
        Arc::clone(&schema),
    );
    let right = builder.add_limit(right_sort, LimitSpec::new(15, Some(10)), Arc::clone(&schema));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("left_key", DataType::Int64, false),
        Field::new("right_key", DataType::Int64, false),
        Field::new("left_sum", DataType::Int64, true),
        Field::new("right_sum", DataType::Int64, true),
        Field::new("left_avg", DataType::Float64, true),
        Field::new("right_avg", DataType::Float64, true),
    ]));
    let joined = builder.add_join(
        left,
        right,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "count_value", "count_value"),
                ColumnRef::input(1, "count_value", "count_value"),
            )],
        )
        .with_build_side(JoinSide::Right)
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "key", "key"), "left_key"),
            JoinOutputColumn::right(ColumnRef::input(1, "key", "key"), "right_key"),
            JoinOutputColumn::left(ColumnRef::input(0, "sum_value", "sum_value"), "left_sum"),
            JoinOutputColumn::right(ColumnRef::input(1, "sum_value", "sum_value"), "right_sum"),
            JoinOutputColumn::left(ColumnRef::input(0, "avg_value", "avg_value"), "left_avg"),
            JoinOutputColumn::right(ColumnRef::input(1, "avg_value", "avg_value"), "right_avg"),
        ]),
        output_schema,
    );
    let plan = builder.finish_with_host_sink(joined);
    let pipeline = compile_pipeline(&plan, &plan.capability_analysis(rapids_analysis_context())).unwrap();
    let scheduler = CompiledPipelineScheduler::new(&pipeline).unwrap();
    let cap = 960 * 1024 * 1024;
    let context = backend_context_with_memory_control_policy(MemoryControlPolicy {
        device_budget_bytes: cap,
        max_source_chunk_bytes: 8 * 1024 * 1024,
        max_row_groups_per_chunk: 1,
        max_retry_attempts: 3,
        min_retry_chunk_row_groups: 1,
    });
    let admitted = context.admit_plan_for_compiled_executor_test(&plan).unwrap();
    let gpu_inputs = GpuResidentInputMap::new();
    let metrics_recorder = ExecutionMetrics::detached("metrics");
    let (state, batch) = {
        let metrics_context = metrics_recorder.scope_context(&admitted).unwrap();
        let mut executor = CompiledGpuTaskExecutor::new(
            &metrics_context,
            &plan,
            PlanningPolicy::default(),
            &pipeline,
            &gpu_inputs,
            &metrics_recorder,
        )
        .unwrap();
        let state = scheduler.run(&mut executor).unwrap();
        (state, executor.take_host_output().unwrap())
    };
    assert_eq!(state.live_resource_count(), 0);
    assert_eq!(batch.num_rows(), 100);
    let mut pairs = std::collections::BTreeSet::new();
    for row in 0..batch.num_rows() {
        let left = int64_array(&batch, 0).value(row);
        let right = int64_array(&batch, 1).value(row);
        assert_eq!(int64_array(&batch, 2).value(row), left % 7);
        assert_eq!(int64_array(&batch, 3).value(row), right % 7);
        assert_eq!(float64_array(&batch, 4).value(row), (left % 7) as f64);
        assert_eq!(float64_array(&batch, 5).value(row), (right % 7) as f64);
        pairs.insert((left, right));
    }
    let expected = (5..15).flat_map(|left| (15..25).map(move |right| (left, right))).collect();
    assert_eq!(pairs, expected);
    let usage = admitted.attempt_allocation_domain_usage().unwrap().unwrap();
    assert!(usage.peak_bytes <= cap, "{usage:?}");
    assert_eq!((usage.current_bytes, usage.current_allocations), (0, 0), "{usage:?}");
    admitted.finish();
    drop(admitted);

    let admitted = context.admit_plan_for_compiled_executor_test(&plan).unwrap();
    let cancellation_metrics_recorder = ExecutionMetrics::detached("cancellation_metrics");
    let (result, retained_at_cancel) = {
        let cancellation_metrics_context = cancellation_metrics_recorder.scope_context(&admitted).unwrap();
        let mut executor = CompiledGpuTaskExecutor::new(
            &cancellation_metrics_context,
            &plan,
            PlanningPolicy::default(),
            &pipeline,
            &gpu_inputs,
            &cancellation_metrics_recorder,
        )
        .unwrap();
        let mut cancel_after_consumer =
            CancelAfterFinalAggregateConsumer { executor: &mut executor, context: &admitted, retained_at_cancel: None };
        let result = scheduler.run(&mut cancel_after_consumer);
        (result, cancel_after_consumer.retained_at_cancel)
    };
    let error = result.err().expect("cancellation stops before the remaining consumer");
    assert_eq!(error.code(), QueryErrorCode::Cancelled);
    let retained = retained_at_cancel.expect("one consumer must finish before cancellation");
    assert!(retained >= 128 * 131_072 * 8, "the remaining consumer still owns all group keys: {retained}");
    assert_attempt_domain_released(admitted.context.context());
    admitted.finish();
    fs::remove_dir_all(dir).unwrap();
}

struct CancelAfterFinalAggregateConsumer<'a, 'execution> {
    executor: &'a mut CompiledGpuTaskExecutor<'execution>,
    context: &'a ExecutionContext,
    retained_at_cancel: Option<u64>,
}

impl CompiledTaskExecutor for CancelAfterFinalAggregateConsumer<'_, '_> {
    type Resource = CompiledGpuResource;

    fn execute_task(
        &mut self,
        task: &PipelineTask,
        state: &mut CompiledPipelineState<Self::Resource>,
    ) -> QueryEngineResult<Vec<(ResidentResourceId, Self::Resource)>> {
        let output = self.executor.execute_task(task, state)?;
        if task.kind() == PipelineTaskKind::FinalSort && self.retained_at_cancel.is_none() {
            self.retained_at_cancel = Some(self.context.attempt_allocation_domain_usage()?.unwrap().current_bytes);
            self.context.cancellation().cancel();
        }
        Ok(output)
    }

    fn release_resource(&mut self, resource: ResidentResourceId, payload: Self::Resource) -> QueryEngineResult<()> {
        self.executor.release_resource(resource, payload)
    }

    fn cleanup_after_error(&mut self) -> QueryEngineResult<()> {
        self.executor.cleanup_after_error()
    }
}
