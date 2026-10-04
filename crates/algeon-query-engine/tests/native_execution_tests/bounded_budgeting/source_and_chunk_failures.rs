use std::sync::Arc;

use query_runtime::runtime::ExecutionCancellation;

use super::*;

#[test]
fn bounded_native_query_enforces_the_query_memory_cap() {
    let dir = unique_temp_dir("query-engine-bounded-reservation-failure");
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
    let aggregate = builder.add_aggregate(source, aggregate, aggregated_schema).unwrap();
    let plan = builder.finish_with_host_sink(aggregate).unwrap();
    let context = backend_context_with_memory_control_policy(MemoryControlPolicy {
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
    assert_eq!(error.wire_fact("budget_bytes").as_deref(), Some("1"));
    assert_eq!(error.wire_fact("budget_enforcement").as_deref(), Some("query_domain_member"));
    assert_eq!(error.fact_value("failure_source"), Some("RESERVATION_LIMIT"));
    assert!(error.fact_value("requested_allocation_bytes").is_some());

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_uses_exact_source_owner_instead_of_spurious_chunk_retries() {
    let dir = unique_temp_dir("query-engine-bounded-chunk-retry-exhausted");
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
    let aggregate = builder.add_aggregate(source_id, aggregate, aggregated_schema).unwrap();
    let plan = builder.finish_with_host_sink(aggregate).unwrap();
    let context = backend_context_with_memory_control_policy(MemoryControlPolicy {
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
    let aggregate = builder.add_aggregate(source, aggregate, output_schema).unwrap();
    let plan = builder.finish_with_host_sink(aggregate).unwrap();
    let context = backend_context_with_memory_control_policy(MemoryControlPolicy {
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
    let dir = unique_temp_dir("query-engine-bounded-chunk-retry");
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
    let filter = builder.add_filter(source_id, filter).unwrap();
    let projection = builder.add_projection(filter, projection_outputs(projection, projected_schema)).unwrap();
    let aggregate = builder.add_aggregate(projection, aggregate, Arc::clone(&aggregated_schema)).unwrap();
    let sort = builder.add_sort(aggregate, sort).unwrap();
    let limit = builder.add_limit(sort, LimitSpec::fetch(3)).unwrap();
    let plan = builder.finish_with_host_sink(limit).unwrap();

    let context = backend_context_with_memory_control_policy(MemoryControlPolicy {
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
    let dir = unique_temp_dir("query-engine-bounded-source-only-retry-order");
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
    let plan = builder.finish_with_host_sink(source_id).unwrap();

    let context = backend_context_with_memory_control_policy(MemoryControlPolicy {
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

// Real query-domain cap recovery: every failure below is produced by the
// allocator rejecting an allocation against the immutable attempt cap, never by
// an injected error code.

const CAP_RECOVERY_ROWS_PER_ROW_GROUP: usize = 65_536;

fn cap_recovery_plan(path: &Path, input_schema: &Arc<Schema>) -> QueryPlan {
    let aggregated_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("sum_value", DataType::Int64, true),
    ]));
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(path).with_file_size_bytes(fs::metadata(path).unwrap().len())],
        Arc::clone(input_schema),
        vec!["key".to_string(), "value".to_string()],
    );
    let aggregate = AggregateSpec::new(
        vec![Expr::column("key")],
        vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("value"), "sum_value")],
    );
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let aggregate = builder.add_aggregate(source, aggregate, aggregated_schema).unwrap();
    builder.finish_with_host_sink(aggregate).unwrap()
}

fn cap_recovery_policy(device_budget_bytes: u64, max_row_groups_per_chunk: usize) -> MemoryControlPolicy {
    MemoryControlPolicy {
        device_budget_bytes,
        max_source_chunk_bytes: 64 * 1024 * 1024,
        max_row_groups_per_chunk,
        max_retry_attempts: 8,
        min_retry_chunk_row_groups: 1,
    }
}

fn assert_cap_recovery_output(batch: &RecordBatch, rows: usize) {
    let expected = uncompressed_row_grouped_expected_sums(rows);
    assert_eq!(batch.num_rows(), 4);
    let keys = int32_array(batch, 0);
    let sums = int64_array(batch, 1);
    for row in 0..batch.num_rows() {
        assert_eq!(sums.value(row), expected[keys.value(row) as usize], "sum for key {}", keys.value(row));
    }
}

#[test]
fn bounded_native_query_recovers_from_query_cap_rejection_by_splitting_source_chunks() {
    let dir = unique_temp_dir("query-engine-bounded-cap-split-recovery");
    let path = dir.join("input.parquet");
    let row_groups = 8;
    let input_schema = write_uncompressed_row_grouped_input_parquet(&path, row_groups, CAP_RECOVERY_ROWS_PER_ROW_GROUP);
    let plan = cap_recovery_plan(&path, &input_schema);
    // One 8-row-group chunk decodes to 6 MiB, which the reader cannot stage
    // under this cap; halves and quarters fit.
    let context = backend_context_with_memory_control_policy(cap_recovery_policy(5 * 1024 * 1024, row_groups));

    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.source_chunks_planned, 1);
    assert_eq!(metrics.row_groups_committed, row_groups as u64);
    assert!(metrics.source_chunks_retried >= 1, "the whole-chunk read must be split after the cap rejection");
    assert!(metrics.retry_resource_exhausted_count >= 1, "cap rejections must be recorded as recoverable retries");
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_cap_recovery_output(batch, row_groups * CAP_RECOVERY_ROWS_PER_ROW_GROUP);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_fails_with_original_cap_cause_when_no_row_group_fits() {
    let dir = unique_temp_dir("query-engine-bounded-cap-terminal");
    let path = dir.join("input.parquet");
    let row_groups = 8;
    let input_schema = write_uncompressed_row_grouped_input_parquet(&path, row_groups, CAP_RECOVERY_ROWS_PER_ROW_GROUP);
    let plan = cap_recovery_plan(&path, &input_schema);
    // A single row group decodes to 768 KiB; nothing fits under 256 KiB.
    let context = backend_context_with_memory_control_policy(cap_recovery_policy(256 * 1024, row_groups));
    let admitted = context.admit_plan(&plan).unwrap();

    let admitted_plan =
        admit_query_plan(&plan, admitted.statistical_aggregate_execution_mode(), admitted.planning_policy()).unwrap();
    let outcome = execute_query_plan_direct(
        admitted.context.context(),
        QueryExecutionRequest::admitted(&admitted_plan).with_gpu_inputs(&GpuResidentInputMap::new()),
    );
    let (result, metrics, _observation) = outcome.into_parts();
    let metrics = metrics.snapshot();
    let error = result.err().expect("no split can fit a single row group under the cap");
    assert_eq!(error.code(), QueryErrorCode::QueryMemoryLimitExceeded);
    assert_eq!(error.fact_value("failure_source"), Some("RESERVATION_LIMIT"));
    assert_eq!(error.fact_value("retry_action"), Some("fail_min_chunk"));
    assert_eq!(error.wire_fact("resume_boundary").as_deref(), Some("source_chunk"));
    assert!(metrics.retry_resource_exhausted_count >= 1, "each cap rejection before the floor is recorded");
    assert_attempt_domain_released(admitted.context.context());

    admitted.finish();
    drop(admitted);
    wait_for_ledger_release(context.policy);
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_stops_recovery_at_the_next_cancellation_boundary() {
    let dir = unique_temp_dir("query-engine-bounded-cap-cancel");
    let path = dir.join("input.parquet");
    let row_groups = 64;
    let rows_per_row_group = 16_384;
    let input_schema = write_uncompressed_row_grouped_input_parquet(&path, row_groups, rows_per_row_group);
    let plan = cap_recovery_plan(&path, &input_schema);
    let context = backend_context_with_memory_control_policy(cap_recovery_policy(1024 * 1024, row_groups));
    // Warm the reader and kernels so the deadline below elapses inside the
    // recovery tree rather than during first-use initialization.
    let warm = backend_context_with_memory_control_policy(cap_recovery_policy(64 * 1024 * 1024, row_groups));
    let warm_result = execute_query_plan(&warm, &plan).unwrap();
    let QueryOutput::HostArrow(batch) = warm_result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_cap_recovery_output(batch, row_groups * rows_per_row_group);

    // Only one row group fits, so recovery needs six halvings with a backoff
    // between each before the first read can succeed; the deadline elapses in
    // that window and the next boundary must exit with the cancellation cause.
    let cancellation =
        ExecutionCancellation::with_deadline(Some(std::time::Instant::now() + std::time::Duration::from_millis(200)));
    let evidence = capability_evidence_for_fixture_plan(&plan).unwrap();
    let handle = suite_backend_with_policy(context.policy)
        .session_capability()
        .admit_query_with_request(
            QueryAdmissionRequest::test_attempt()
                .with_query_plan_capability_evidence(evidence)
                .with_query_cap_bytes(1024 * 1024),
        )
        .unwrap();
    let admitted = handle.acquire_context_with_cancellation(cancellation).unwrap();
    let admitted_plan = admit_query_plan(
        &plan,
        admitted.context().statistical_aggregate_execution_mode(),
        admitted.context().planning_policy(),
    )
    .unwrap();
    let outcome = execute_query_plan_direct(
        admitted.context(),
        QueryExecutionRequest::admitted(&admitted_plan).with_gpu_inputs(&GpuResidentInputMap::new()),
    );
    let (result, metrics, _observation) = outcome.into_parts();
    let metrics = metrics.snapshot();
    let error = result.err().expect("the deadline elapses before recovery completes");
    assert_eq!(error.code(), QueryErrorCode::Cancelled, "{error:?}");
    assert_eq!(error.fact_value("reason"), Some("deadline_expired"));
    assert!(metrics.retry_resource_exhausted_count >= 1, "cancellation must land inside the recovery tree");
    assert_attempt_domain_released(admitted.context());

    drop(admitted);
    handle.end_test_attempt();
    drop(handle);
    wait_for_ledger_release(context.policy);
    fs::remove_dir_all(dir).unwrap();
}
