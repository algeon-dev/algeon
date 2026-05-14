use super::*;

#[test]
fn bounded_native_query_reports_reservation_failure_before_chunk_read() {
    let dir = unique_temp_dir("cudf-query-engine-bounded-reservation-failure");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
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
    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 1,
            max_source_chunk_bytes: 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .build()
        .unwrap();

    let error = match execute_query_plan(&context, &plan) {
        Ok(_) => panic!("bounded execution should fail the source reservation"),
        Err(error) => error,
    };
    assert_eq!(error.code(), QueryErrorCode::MemoryReservationFailed);
    assert_eq!(error.fact_value("memory_boundary"), Some("source_chunk_read"));
    assert_eq!(error.fact_value("reservation_kind"), Some("source_read"));
    assert_eq!(error.fact_value("budget_bytes"), Some("1"));
    assert_eq!(error.fact_value("chunk_index"), Some("0"));
    assert_eq!(error.fact_value("file_index"), Some("0"));
    assert_eq!(error.fact_value("row_groups"), Some("0"));
    assert_eq!(error.fact_value("stage"), Some("source_chunk_read"));
    assert_eq!(error.fact_value("attempt"), Some("1"));
    assert_eq!(error.fact_value("max_attempts"), Some("1"));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_reports_retry_facts_when_chunk_retry_exhausts() {
    let dir = unique_temp_dir("cudf-query-engine-bounded-chunk-retry-exhausted");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
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
    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 1,
            max_source_chunk_bytes: 1024 * 1024,
            max_row_groups_per_chunk: 3,
            max_retry_attempts: 2,
            min_retry_chunk_row_groups: 1,
        })
        .build()
        .unwrap();

    let error = match execute_query_plan(&context, &plan) {
        Ok(_) => panic!("bounded execution should fail after exhausting source chunk retries"),
        Err(error) => error,
    };

    assert_eq!(error.code(), QueryErrorCode::MemoryReservationFailed);
    assert_eq!(error.fact_value("memory_boundary"), Some("source_chunk_read"));
    assert_eq!(error.fact_value("reservation_kind"), Some("source_read"));
    assert_eq!(error.fact_value("stage"), Some("source_chunk_read"));
    assert_eq!(error.fact_value("chunk_index"), Some("0"));
    assert_eq!(error.fact_value("file_index"), Some("0"));
    assert_eq!(error.fact_value("row_groups"), Some("0"));
    assert_eq!(error.fact_value("attempt"), Some("2"));
    assert_eq!(error.fact_value("max_attempts"), Some("2"));
    assert_eq!(error.fact_value("retry_action"), Some("fail_min_chunk"));
    assert_eq!(error.fact_value("resume_boundary"), Some("source_chunk"));
    assert_eq!(error.fact_value("committed_chunks"), Some("0"));
    assert_eq!(error.fact_value("committed_partials"), Some("0"));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_reports_source_chunk_hard_budget_failure_after_reservation_admission() {
    let dir = unique_temp_dir("cudf-query-engine-bounded-source-hard-budget-failure");
    let path = dir.join("input.parquet");
    let input_schema = write_single_i32_row_grouped_parquet(&path, 4);
    let aggregated_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("sum_id", DataType::Int64, true),
    ]));
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["id".to_string()],
    );
    let aggregate = AggregateSpec::new(
        vec![Expr::column("id")],
        vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("id"), "sum_id")],
    );

    let mut builder = QueryPlanBuilder::new();
    let source_id = builder.add_source(SourceNode::LocalParquet(source.clone()));
    let aggregate = builder.add_aggregate(source_id, aggregate, aggregated_schema);
    let plan = builder.finish_with_host_sink(aggregate);
    let chunk_plan = plan_local_parquet_source_chunks(source_id, &source, SourceChunkPlanningPolicy::new(None, 1))
        .expect("test source should produce a planned row-group chunk");
    let admitted_budget = chunk_plan.chunks[0]
        .known_row_group_bytes
        .expect("test parquet metadata should expose row-group bytes")
        .saturating_add(1);
    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::new(cudf::Runtime::new().unwrap()))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: admitted_budget,
            max_source_chunk_bytes: 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .build()
        .unwrap();

    let error = match execute_query_plan(&context, &plan) {
        Ok(_) => panic!("hard cudf source read budget should fail"),
        Err(error) => error,
    };
    let admitted_budget = admitted_budget.to_string();
    assert_eq!(error.code(), QueryErrorCode::CudfResourceExhausted);
    assert_eq!(error.fact_value("budget_boundary"), Some("source_chunk_read"));
    assert_eq!(error.fact_value("budget_bytes"), Some(admitted_budget.as_str()));
    assert_eq!(error.fact_value("budget_enforcement"), Some("hard_memory_resource_limit"));
    assert_eq!(error.fact_value("budget_status"), Some("exceeded_or_lower_layer_rejected"));
    assert!(error.fact_value("budget_peak_bytes").is_some());
    assert!(error.fact_value("budget_current_bytes").is_some());
    assert!(error.fact_value("budget_failed_allocation_count").is_some());
    assert!(error.fact_value("budget_largest_failed_allocation_bytes").is_some());
    assert_eq!(error.fact_value("memory_boundary"), Some("source_chunk_read"));
    assert_eq!(error.fact_value("stage"), Some("source_chunk_read"));
    assert_eq!(error.fact_value("chunk_index"), Some("0"));
    assert_eq!(error.fact_value("file_index"), Some("0"));
    assert_eq!(error.fact_value("row_groups"), Some("0"));
    assert_eq!(error.fact_value("attempt"), Some("1"));
    assert_eq!(error.fact_value("max_attempts"), Some("1"));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_reports_chunk_operator_hard_budget_failure_after_reservation_admission() {
    let dir = unique_temp_dir("cudf-query-engine-bounded-chunk-operator-hard-budget-failure");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_final_merge_retry_input_parquet(&path, 512);
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
            .collect(),
    );

    let mut builder = QueryPlanBuilder::new();
    let source_id = builder.add_source(SourceNode::LocalParquet(source));
    let aggregate = builder.add_aggregate(source_id, aggregate, aggregated_schema);
    let plan = builder.finish_with_host_sink(aggregate);

    let baseline_context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 8 * 1024 * 1024 * 1024,
            max_source_chunk_bytes: 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .build()
        .unwrap();
    let baseline = execute_query_plan(&baseline_context, &plan).unwrap();
    let mut source_budget = 0_u64;
    let mut operator_budget = None;
    for event in &baseline.metrics().reservation_events {
        if event.outcome.as_str() != "granted" {
            continue;
        }
        match (event.reservation_kind, event.memory_boundary.as_str()) {
            (ReservationKind::SourceRead, "source_chunk_read") => {
                source_budget = source_budget.saturating_add(event.requested_bytes);
            },
            (ReservationKind::ChunkOperator, "chunk_operator") => {
                operator_budget = Some(event.requested_bytes);
                break;
            },
            _ => {},
        }
    }
    let operator_budget = operator_budget.expect("baseline should reserve chunk operator work");
    assert!(source_budget > 0);
    assert!(operator_budget > 0);

    let operator_token_budget = operator_budget.saturating_mul(4);
    let constrained_context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: source_budget.saturating_add(operator_token_budget),
            max_source_chunk_bytes: 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .build()
        .unwrap();

    let error = match execute_query_plan(&constrained_context, &plan) {
        Ok(_) => panic!("hard cudf chunk operator budget should fail"),
        Err(error) => error,
    };
    let operator_budget = operator_token_budget.to_string();
    assert_eq!(error.code(), QueryErrorCode::CudfResourceExhausted);
    assert_eq!(error.fact_value("budget_boundary"), Some("chunk_operator"));
    assert_eq!(error.fact_value("budget_bytes"), Some(operator_budget.as_str()));
    assert_eq!(error.fact_value("budget_enforcement"), Some("hard_memory_resource_limit"));
    assert_eq!(error.fact_value("memory_boundary"), Some("chunk_operator"));
    assert_eq!(error.fact_value("stage"), Some("chunk_operator"));
    assert_eq!(error.fact_value("chunk_index"), Some("0"));
    assert_eq!(error.fact_value("file_index"), Some("0"));
    assert_eq!(error.fact_value("row_groups"), Some("0"));
    assert_eq!(error.fact_value("attempt"), Some("1"));
    assert_eq!(error.fact_value("max_attempts"), Some("1"));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_does_not_prefetch_before_first_required_source_read_is_observed() {
    let dir = unique_temp_dir("cudf-query-engine-bounded-initial-prefetch-calibration");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_row_grouped_input_parquet(&path);
    let aggregated_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("sum_m00", DataType::Int64, true),
    ]));
    let source = LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new(&path)
                .with_file_size_bytes(fs::metadata(&path).unwrap().len())
                .with_row_group_bytes(vec![1, 1, 1])
                .with_projected_row_group_bytes(vec![1, 1, 1], vec![1, 1, 1])
                .with_row_group_rows(vec![2, 2, 2]),
        ],
        Arc::clone(&input_schema),
        vec!["key".to_string(), "m00".to_string()],
    );
    let aggregate = AggregateSpec::new(
        vec![Expr::column("key")],
        vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("m00"), "sum_m00")],
    );

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let aggregate = builder.add_aggregate(source, aggregate, Arc::clone(&aggregated_schema));
    let plan = builder.finish_with_host_sink(aggregate);
    let max_source_chunk_bytes = 1024 * 1024;
    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: max_source_chunk_bytes + 1,
            max_source_chunk_bytes,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .with_source_scheduler_policy(SourceSchedulerPolicy::new(2))
        .build()
        .unwrap();

    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.source_chunks_planned, 3);
    assert_eq!(metrics.source_chunks_committed, 3);
    assert_eq!(metrics.source_read_prefetches_started, 0);
    assert!(metrics.source_read_prefetches_skipped >= 1);
    assert_eq!(metrics.reservation_denied_count, 0);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 3);
    let mut rows = int32_array(batch, 0)
        .values()
        .iter()
        .copied()
        .zip(int64_array(batch, 1).values().iter().copied())
        .collect::<Vec<_>>();
    rows.sort_by_key(|(key, _sum)| *key);
    assert_eq!(rows, vec![(1, 400), (2, 700), (3, 1000)]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_reports_global_count_scalar_output_hard_budget_failure() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
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
    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 1024,
            max_source_chunk_bytes: 1,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .build()
        .unwrap();

    let error = match execute_query_plan(&context, &plan) {
        Ok(_) => panic!("bounded scalar-to-series output should fail the hard operator budget"),
        Err(error) => error,
    };
    assert_eq!(error.code(), QueryErrorCode::CudfResourceExhausted);
    assert_eq!(error.operation(), "cudf_query_engine.global_aggregate.count_output");
    assert_eq!(error.fact_value("budget_boundary"), Some("chunk_operator"));
    assert_eq!(error.fact_value("budget_bytes"), Some("1016"));
    assert_eq!(error.fact_value("budget_enforcement"), Some("hard_memory_resource_limit"));
    assert_eq!(error.fact_value("budget_operation"), Some("runtime::series_from_value_with_budget"));
    assert_eq!(error.fact_value("memory_boundary"), Some("chunk_operator"));
    assert_eq!(error.fact_value("stage"), Some("chunk_operator"));
}

#[test]
fn bounded_native_query_retries_chunk_after_source_reservation_failure() {
    let dir = unique_temp_dir("cudf-query-engine-bounded-chunk-retry");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
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
                .with_row_group_rows(vec![2, 2, 2]),
        ],
        Arc::clone(&input_schema),
        vec!["key".to_string(), "m00".to_string(), "m01".to_string(), "payload".to_string()],
    );
    let filter = Expr::binary(Expr::column("payload"), BinaryOp::GtEq, Expr::literal(LiteralValue::Int64(10)));
    let projection =
        vec![Expr::column("key"), Expr::binary(Expr::column("m00"), BinaryOp::Add, Expr::column("m01")).alias("x00")];
    let aggregate = AggregateSpec::new(
        vec![Expr::column("key")],
        vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("x00"), "sum_x00")],
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
    let initial_source_read_bytes = one_chunk_plan.chunks[0].estimated_source_read_bytes.unwrap_or(initial_chunk_bytes);
    let filter = builder.add_filter(source_id, filter, Arc::clone(&input_schema));
    let projection = builder.add_projection(filter, projection_outputs(projection, projected_schema));
    let aggregate = builder.add_aggregate(projection, aggregate, Arc::clone(&aggregated_schema));
    let sort = builder.add_sort(aggregate, sort, Arc::clone(&aggregated_schema));
    let limit = builder.add_limit(sort, LimitSpec::fetch(3), aggregated_schema);
    let plan = builder.finish_with_host_sink(limit);

    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: initial_source_read_bytes.saturating_sub(1),
            max_source_chunk_bytes: initial_chunk_bytes,
            max_row_groups_per_chunk: 3,
            max_retry_attempts: 3,
            min_retry_chunk_row_groups: 1,
        })
        .build()
        .unwrap();

    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.source_chunks_planned, 1);
    assert_eq!(metrics.source_chunks_retried, 2);
    assert_eq!(metrics.retry_count, 2);
    assert_eq!(metrics.retry_reservation_failure_count, 1);
    assert_eq!(metrics.retry_resource_exhausted_count, 1);
    assert_eq!(metrics.reservation_denied_count, 1);
    assert_reservation_events_memory_space(metrics, MemorySpaceId::Device { device_ordinal: 0 });
    assert_eq!(metrics.source_chunks_attempted, 5);
    assert_eq!(metrics.source_chunks_committed, 3);
    assert_eq!(metrics.row_groups_committed, 3);
    assert_eq!(metrics.partial_aggregate_count, 3);
    assert_eq!(metrics.final_merge_attempts, 1);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 3);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[2, 1, 3]);
    assert_eq!(int64_array(batch, 1).values().as_ref(), &[707, 404, 404]);

    fs::remove_dir_all(dir).unwrap();
}
