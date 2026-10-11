use std::sync::Arc;

use super::*;

#[path = "../../../support/incremental_merge_fixture.rs"]
mod incremental_merge_fixture;

use incremental_merge_fixture::incremental_partitioned_merge_builder;

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
    let aggregate = builder.add_aggregate(source, aggregate, Arc::clone(&aggregated_schema)).unwrap();
    let plan = builder.finish_with_host_sink(aggregate).unwrap();

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

fn incremental_partitioned_merge_plans(
    dir: &Path,
    row_groups: usize,
    rows_per_row_group: usize,
) -> (QueryPlan, QueryPlan) {
    let (mut builder, aggregate, _) =
        incremental_partitioned_merge_builder(dir, row_groups, rows_per_row_group, LocalParquetSource::from_files);
    let plan = builder.clone().finish_with_host_sink(aggregate).unwrap();
    let sort =
        builder.add_sort(aggregate, SortSpec::new(vec![SortExpr::new(Expr::column("key"), false, false)])).unwrap();
    let limit = builder.add_limit(sort, LimitSpec::new(5, Some(10))).unwrap();
    (plan, builder.finish_with_host_sink(limit).unwrap())
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
    let metrics = metrics.snapshot();
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
fn bounded_grouped_aggregate_splits_wide_partials_without_resource_retries() {
    let dir = unique_temp_dir("query-engine-bounded-proactive-partial-split");
    let path = dir.join("input.parquet");
    let rows_per_row_group = 1024;
    let rows = 3 * rows_per_row_group;
    let input_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("m00", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&input_schema),
        vec![
            Arc::new(Int32Array::from((0..rows as i32).map(|row| row % 4).collect::<Vec<_>>())),
            Arc::new(Int64Array::from(vec![1_i64; rows])),
        ],
    )
    .unwrap();
    write_record_batch_with_row_group_rows(&path, Arc::clone(&input_schema), batch, rows_per_row_group);
    let repeated_sums = 256;
    let mut fields = vec![Field::new("key", DataType::Int32, false)];
    fields.extend((0..repeated_sums).map(|index| Field::new(format!("sum_m00_{index}"), DataType::Int64, true)));
    let aggregated_schema = Arc::new(Schema::new(fields));
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["key".to_string(), "m00".to_string()],
    );
    let aggregate = AggregateSpec::new(
        vec![Expr::column("key")],
        (0..repeated_sums)
            .map(|index| AggregateExpr::new(AggregateFunction::Sum, Expr::column("m00"), format!("sum_m00_{index}")))
            .collect::<Vec<_>>(),
    );
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let aggregate = builder.add_aggregate(source, aggregate, Arc::clone(&aggregated_schema)).unwrap();
    let plan = builder.finish_with_host_sink(aggregate).unwrap();

    // Each 1024-row chunk bounds its 256-sum partial state at about 2.1 MB,
    // above this cap, so every chunk splits once before allocating. The four
    // groups keep the real partials far below the cap.
    let context = backend_context_with_memory_control_policy(MemoryControlPolicy {
        device_budget_bytes: 1_500_000,
        max_source_chunk_bytes: 1024 * 1024,
        max_row_groups_per_chunk: 1,
        max_retry_attempts: 3,
        min_retry_chunk_row_groups: 1,
    });

    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.source_chunks_planned, 3);
    assert_eq!(metrics.partial_aggregate_count, 6, "each chunk splits into two partials");
    assert_eq!(metrics.retry_count, 0);
    assert_eq!(metrics.retry_resource_exhausted_count, 0, "a proactive split is not an allocation failure");

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 4);
    for index in 1..=repeated_sums {
        assert!(int64_array(batch, index).iter().all(|sum| sum == Some(768)));
    }

    fs::remove_dir_all(dir).unwrap();
}
