use std::sync::Arc;

use super::*;

#[test]
fn bounded_native_query_enforces_the_query_memory_cap() {
    let dir = unique_temp_dir("nexus-query-engine-bounded-reservation-failure");
    let path = dir.join("input.parquet");
    let input_schema = write_row_grouped_input_parquet(&path);
    let aggregated_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("sum_m00", DataType::Int64, true),
    ]));
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["key".to_string(), "m00".to_string()],
    );
    let aggregate = AggregateSpec::new(
        vec![Expr::column("key")],
        vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("m00"), "sum_m00")],
    );

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let aggregate = builder.add_aggregate(source, aggregate, aggregated_schema);
    let plan = builder.finish_with_host_sink(aggregate);
    let context = backend_context_with_memory_control_policy(MemoryControlPolicy::Bounded {
        device_budget_bytes: 1,
        max_source_chunk_bytes: 1024 * 1024,
        max_row_groups_per_chunk: 1,
        max_retry_attempts: 1,
        min_retry_chunk_row_groups: 1,
    });

    let error = match execute_query_plan(&context, &plan) {
        Ok(_) => panic!("bounded execution should enforce the query memory cap"),
        Err(error) => error,
    };
    assert_eq!(error.code(), QueryErrorCode::QueryMemoryLimitExceeded);
    assert_eq!(error.fact_value("budget_bytes"), Some("1"));
    assert_eq!(error.fact_value("budget_enforcement"), Some("query_domain_member"));
    assert_eq!(error.fact_value("failure_source"), Some("RESERVATION_LIMIT"));
    assert!(error.fact_value("requested_allocation_bytes").is_some());

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_uses_exact_source_owner_instead_of_spurious_chunk_retries() {
    let dir = unique_temp_dir("nexus-query-engine-bounded-chunk-retry-exhausted");
    let path = dir.join("input.parquet");
    let input_schema = write_row_grouped_input_parquet(&path);
    let aggregated_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("sum_m00", DataType::Int64, true),
    ]));
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["key".to_string(), "m00".to_string()],
    );
    let aggregate = AggregateSpec::new(
        vec![Expr::column("key")],
        vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("m00"), "sum_m00")],
    );

    let mut builder = QueryPlanBuilder::new();
    let source_id = builder.add_source(SourceNode::LocalParquet(source));
    let aggregate = builder.add_aggregate(source_id, aggregate, aggregated_schema);
    let plan = builder.finish_with_host_sink(aggregate);
    let context = backend_context_with_memory_control_policy(MemoryControlPolicy::Bounded {
        device_budget_bytes: NATIVE_EXECUTION_TEST_QUERY_LIMIT_BYTES,
        max_source_chunk_bytes: 1024 * 1024,
        max_row_groups_per_chunk: 3,
        max_retry_attempts: 2,
        min_retry_chunk_row_groups: 1,
    });

    let result = execute_query_plan(&context, &plan).unwrap();
    assert_eq!(result.metrics().retry_count, 0);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_grants_global_count_source_and_output_owners() {
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("count_rows_0", DataType::Int64, false),
        Field::new("count_rows_1", DataType::Int64, false),
        Field::new("count_rows_2", DataType::Int64, false),
        Field::new("count_rows_3", DataType::Int64, false),
    ]));
    let source = LocalParquetSource::zero_column_count(vec![
        LocalParquetFile::new("/tmp/count-only-budgeted-output.parquet")
            .with_stable_id("count-only-budgeted-output")
            .with_file_size_bytes(0)
            .with_row_groups(vec![0])
            .with_row_group_bytes(vec![0])
            .with_row_group_rows(vec![4096]),
    ]);
    let aggregate = AggregateSpec::new(
        Vec::new(),
        (0..4)
            .map(|index| {
                AggregateExpr::new(
                    AggregateFunction::Count,
                    Expr::literal(LiteralValue::Int64(1)),
                    format!("count_rows_{index}"),
                )
            })
            .collect(),
    );

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let aggregate = builder.add_aggregate(source, aggregate, output_schema);
    let plan = builder.finish_with_host_sink(aggregate);
    let context = backend_context_with_memory_control_policy(MemoryControlPolicy::Bounded {
        device_budget_bytes: NATIVE_EXECUTION_TEST_QUERY_LIMIT_BYTES,
        max_source_chunk_bytes: 1,
        max_row_groups_per_chunk: 1,
        max_retry_attempts: 1,
        min_retry_chunk_row_groups: 1,
    });

    execute_query_plan(&context, &plan).unwrap();
}

#[test]
fn bounded_native_query_does_not_retry_from_source_estimate() {
    let dir = unique_temp_dir("nexus-query-engine-bounded-chunk-retry");
    let path = dir.join("input.parquet");
    let input_schema = write_row_grouped_input_parquet(&path);

    let projected_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("x00", DataType::Int64, false),
    ]));
    let aggregated_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("sum_x00", DataType::Int64, true),
    ]));

    let source = LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new(&path)
                .with_file_size_bytes(fs::metadata(&path).unwrap().len())
                .with_row_group_bytes(vec![4096, 4096, 4096])
                .with_projected_row_group_bytes(vec![4096, 4096, 4096], vec![4096, 4096, 4096])
                .with_row_group_rows(vec![4096, 4096, 4096]),
        ],
        Arc::clone(&input_schema),
        vec!["key".to_string(), "m00".to_string(), "m01".to_string(), "payload".to_string()],
    );
    let filter = Expr::binary(Expr::column("payload"), BinaryOp::GtEq, Expr::literal(LiteralValue::Int64(10)));
    let projection =
        vec![Expr::column("key"), Expr::binary(Expr::column("m00"), BinaryOp::Add, Expr::column("m01")).alias("x00")];
    let aggregate_filter = Expr::binary(Expr::column("x00"), BinaryOp::GtEq, Expr::literal(LiteralValue::Int64(0)));
    let aggregate = AggregateSpec::new(
        vec![Expr::column("key")],
        vec![
            AggregateExpr::new(AggregateFunction::Sum, Expr::column("x00"), "sum_x00")
                .with_filter(Some(aggregate_filter)),
        ],
    );
    let sort = SortSpec::new(vec![
        SortExpr::new(Expr::column("sum_x00"), true, false),
        SortExpr::new(Expr::column("key"), false, false),
    ]);

    let mut builder = QueryPlanBuilder::new();
    let source_id = builder.add_source(SourceNode::LocalParquet(source.clone()));
    let one_chunk_plan =
        plan_local_parquet_source_chunks(source_id, &source, SourceChunkPlanningPolicy::new(None, 3)).unwrap();
    assert_eq!(one_chunk_plan.chunk_count(), 1);
    let initial_chunk_bytes = one_chunk_plan.chunks[0].known_row_group_bytes.unwrap();
    let filter = builder.add_filter(source_id, filter, Arc::clone(&input_schema));
    let projection = builder.add_projection(filter, projection_outputs(projection, projected_schema));
    let aggregate = builder.add_aggregate(projection, aggregate, Arc::clone(&aggregated_schema));
    let sort = builder.add_sort(aggregate, sort, Arc::clone(&aggregated_schema));
    let limit = builder.add_limit(sort, LimitSpec::fetch(3), aggregated_schema);
    let plan = builder.finish_with_host_sink(limit);

    let context = backend_context_with_memory_control_policy(MemoryControlPolicy::Bounded {
        device_budget_bytes: NATIVE_EXECUTION_TEST_QUERY_LIMIT_BYTES,
        max_source_chunk_bytes: initial_chunk_bytes,
        max_row_groups_per_chunk: 3,
        max_retry_attempts: 3,
        min_retry_chunk_row_groups: 1,
    });

    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.source_chunks_planned, 1);
    assert_eq!(metrics.source_chunks_retried, 0);
    assert_eq!(metrics.retry_count, 0);
    assert_eq!(metrics.retry_resource_exhausted_count, 0);
    assert_eq!(metrics.source_chunks_attempted, 1);
    assert_eq!(metrics.source_chunks_committed, 1);
    assert_eq!(metrics.row_groups_committed, 3);
    assert_eq!(metrics.partial_aggregate_count, 1);
    assert_eq!(metrics.final_merge_attempts, 0);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 3);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[2, 1, 3]);
    assert_eq!(int64_array(batch, 1).values().as_ref(), &[707, 404, 404]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_source_owner_preserves_row_group_order_without_spurious_retry() {
    let dir = unique_temp_dir("nexus-query-engine-bounded-source-only-retry-order");
    let path = dir.join("input.parquet");
    let input_schema = write_row_grouped_input_parquet(&path);
    let source = LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new(&path)
                .with_file_size_bytes(fs::metadata(&path).unwrap().len())
                .with_row_group_bytes(vec![4096, 4096, 4096])
                .with_projected_row_group_bytes(vec![4096, 4096, 4096], vec![4096, 4096, 4096])
                .with_row_group_rows(vec![2, 2, 2]),
        ],
        Arc::clone(&input_schema),
        vec!["key".to_string(), "payload".to_string(), "m00".to_string(), "m01".to_string()],
    );

    let mut builder = QueryPlanBuilder::new();
    let source_id = builder.add_source(SourceNode::LocalParquet(source.clone()));
    let one_chunk_plan =
        plan_local_parquet_source_chunks(source_id, &source, SourceChunkPlanningPolicy::new(None, 3)).unwrap();
    assert_eq!(one_chunk_plan.chunk_count(), 1);
    let initial_chunk_bytes = one_chunk_plan.chunks[0].known_row_group_bytes.unwrap();
    let plan = builder.finish_with_host_sink(source_id);

    let context = backend_context_with_memory_control_policy(MemoryControlPolicy::Bounded {
        device_budget_bytes: NATIVE_EXECUTION_TEST_QUERY_LIMIT_BYTES,
        max_source_chunk_bytes: initial_chunk_bytes,
        max_row_groups_per_chunk: 3,
        max_retry_attempts: 3,
        min_retry_chunk_row_groups: 1,
    });

    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.source_chunks_planned, 1);
    assert_eq!(metrics.source_chunks_retried, 0);
    assert_eq!(metrics.retry_count, 0);
    assert_eq!(metrics.source_chunks_attempted, 1);
    assert_eq!(metrics.source_chunks_committed, 1);
    assert_eq!(metrics.row_groups_committed, 3);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 6);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[1, 2, 1, 3, 2, 3]);
    assert_eq!(int64_array(batch, 1).values().as_ref(), &[10, 10, 20, 20, 30, 5]);

    fs::remove_dir_all(dir).unwrap();
}
