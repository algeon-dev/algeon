use std::sync::Arc;

use super::*;

#[test]
fn bounded_native_query_executes_global_final_merge_with_consistent_state_schema() {
    let dir = unique_temp_dir("nexus-query-engine-bounded-global-final-merge-retry");
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

    let context = backend_context_with_memory_control_policy(MemoryControlPolicy::Bounded {
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
        batch.schema().fields().iter().all(|field| !field.name().starts_with("__nexus_native_global_valid_count_"))
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
    let dir = unique_temp_dir("nexus-query-engine-raw-bounded-final-merge-retry");
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

    let context = backend_context_with_memory_control_policy(MemoryControlPolicy::Bounded {
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

#[test]
fn bounded_native_query_merges_high_cardinality_partials_incrementally_under_the_query_cap() {
    let dir = unique_temp_dir("nexus-query-engine-bounded-incremental-partitioned-merge");
    let path = dir.join("input.parquet");
    let row_groups = 32;
    let rows_per_row_group = 131_072;
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
            vec![JoinKey::new(
                nexus_query_engine::expr::ColumnRef::input(0, "d", "d"),
                nexus_query_engine::expr::ColumnRef::input(1, "d", "d"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(nexus_query_engine::expr::ColumnRef::input(0, "key", "key"), "key"),
            JoinOutputColumn::left(nexus_query_engine::expr::ColumnRef::input(0, "value", "value"), "value"),
        ])
        .with_build_side(JoinSide::Right),
        Arc::clone(&join_schema),
    );
    let aggregate = builder.add_aggregate(join, aggregate, aggregated_schema);
    let plan = builder.finish_with_host_sink(aggregate);

    // Every row group keeps all of its rows as partial state (about 160 MiB in
    // total across SUM, COUNT, and AVG state). Retaining every partial beside
    // its hash-partition slices and then the merged partitions beside their
    // concatenation needs three times the merged state; folding partitions
    // incrementally keeps the retained set at the merged state plus its final
    // concatenation.
    let context = backend_context_with_memory_control_policy(MemoryControlPolicy::Bounded {
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

    fs::remove_dir_all(dir).unwrap();
}
