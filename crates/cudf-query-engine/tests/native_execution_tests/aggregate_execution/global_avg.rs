use super::*;

#[test]
fn bounded_native_query_executes_global_avg_from_sum_count_state() {
    let dir = unique_temp_dir("cudf-query-engine-bounded-native-global-avg");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_uneven_global_avg_input_parquet(&path);

    let aggregated_schema = Arc::new(Schema::new(vec![Field::new("avg_m00", DataType::Float64, true)]));
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["m00".to_string()],
    );
    let aggregate = AggregateSpec::new(
        Vec::new(),
        vec![AggregateExpr::new(AggregateFunction::Avg, Expr::column("m00"), "avg_m00")],
    );

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let aggregate = builder.add_aggregate(source, aggregate, aggregated_schema);
    let plan = builder.finish_with_host_sink(aggregate);

    let context = ExecutionContext::builder()
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
    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.source_chunks_planned, 2);
    assert_eq!(metrics.source_chunks_committed, 2);
    assert_eq!(metrics.partial_aggregate_count, 2);
    assert_eq!(metrics.final_merge_attempts, 1);
    assert_eq!(metrics.boundary.host_to_device_uploads, 0);
    assert_eq!(metrics.boundary.device_to_host_materializations, 1);
    assert_budgeted_success_event_with_rows(metrics, "cudf_query_engine.aggregate.finalize_avg", 1);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 1);
    assert_eq!(float64_array(batch, 0).value(0), 72.0);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_executes_global_avg_with_wide_sum_state_for_narrow_integer_input() {
    let dir = unique_temp_dir("cudf-query-engine-bounded-native-global-avg-int8");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_narrow_global_avg_input_parquet(&path);

    let aggregated_schema = Arc::new(Schema::new(vec![Field::new("avg_m00", DataType::Float64, true)]));
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["m00".to_string()],
    );
    let aggregate = AggregateSpec::new(
        Vec::new(),
        vec![AggregateExpr::new(AggregateFunction::Avg, Expr::column("m00"), "avg_m00")],
    );

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let aggregate = builder.add_aggregate(source, aggregate, aggregated_schema);
    let plan = builder.finish_with_host_sink(aggregate);

    let context = ExecutionContext::builder()
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
    let result = execute_query_plan(&context, &plan).unwrap();
    assert_eq!(result.metrics().source_chunks_planned, 2);
    assert_eq!(result.metrics().partial_aggregate_count, 2);
    assert_eq!(result.metrics().final_merge_attempts, 1);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 1);
    assert_eq!(float64_array(batch, 0).value(0), 100.0);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_executes_global_null_on_empty_sum_avg_for_all_null_input() {
    let dir = unique_temp_dir("cudf-query-engine-bounded-native-global-all-null");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_nullable_global_aggregate_input_parquet(&path);

    let plan = nullable_global_aggregate_plan(&path, Arc::clone(&input_schema), None);
    let context = ExecutionContext::builder()
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
    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.source_chunks_planned, 2);
    assert_eq!(metrics.source_chunks_committed, 2);
    assert_eq!(metrics.partial_aggregate_count, 2);
    assert_eq!(metrics.final_merge_attempts, 1);
    assert_budgeted_success_event_with_contract(
        metrics,
        "cudf_query_engine.global_aggregate.null_output",
        1,
        "final_aggregate_merge",
        None,
    );

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 1);
    assert!(int64_array(batch, 0).is_null(0));
    assert!(float64_array(batch, 1).is_null(0));
    assert!(int64_array(batch, 2).is_null(0));
    assert!(int64_array(batch, 3).is_null(0));
    assert_eq!(int64_array(batch, 4).value(0), 0);
    assert_eq!(int64_array(batch, 5).value(0), 4);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_executes_global_null_on_empty_sum_avg_for_filtered_empty_input() {
    let dir = unique_temp_dir("cudf-query-engine-bounded-native-global-filtered-empty");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_nullable_global_aggregate_input_parquet(&path);
    let filter = Expr::binary(Expr::column("payload"), BinaryOp::Gt, Expr::literal(LiteralValue::Int64(100)));

    let plan = nullable_global_aggregate_plan(&path, Arc::clone(&input_schema), Some(filter));
    let context = ExecutionContext::builder()
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
    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.source_chunks_planned, 2);
    assert_eq!(metrics.source_chunks_committed, 2);
    assert_eq!(metrics.partial_aggregate_count, 2);
    assert_eq!(metrics.final_merge_attempts, 1);
    assert_budgeted_success_event_with_contract(
        metrics,
        "cudf_query_engine.global_aggregate.null_output",
        1,
        "final_aggregate_merge",
        None,
    );

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 1);
    assert!(int64_array(batch, 0).is_null(0));
    assert!(float64_array(batch, 1).is_null(0));
    assert!(int64_array(batch, 2).is_null(0));
    assert!(int64_array(batch, 3).is_null(0));
    assert_eq!(int64_array(batch, 4).value(0), 0);
    assert_eq!(int64_array(batch, 5).value(0), 0);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_executes_decimal_global_avg_from_sum_count_state() {
    let dir = unique_temp_dir("cudf-query-engine-bounded-native-decimal-global-avg");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_decimal_avg_input_parquet(&path);

    let aggregated_schema = Arc::new(Schema::new(vec![Field::new("avg_amount", DataType::Decimal128(19, 6), true)]));
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["amount".to_string()],
    );
    let aggregate = AggregateSpec::new(
        Vec::new(),
        vec![AggregateExpr::new(AggregateFunction::Avg, Expr::column("amount"), "avg_amount")],
    );

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let aggregate = builder.add_aggregate(source, aggregate, aggregated_schema);
    let plan = builder.finish_with_host_sink(aggregate);

    let context = ExecutionContext::builder()
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
    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.source_chunks_planned, 3);
    assert_eq!(metrics.source_chunks_committed, 3);
    assert_eq!(metrics.partial_aggregate_count, 3);
    assert_eq!(metrics.final_merge_attempts, 1);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 1);
    let avg_amount = decimal128_array(batch, 0);
    assert_eq!(avg_amount.value(0), 2_000_000);
    assert_eq!(avg_amount.precision(), 19);
    assert_eq!(avg_amount.scale(), 6);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_executes_fractional_decimal_global_avg_at_output_scale() {
    let dir = unique_temp_dir("cudf-query-engine-bounded-native-fractional-decimal-global-avg");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_fractional_decimal_avg_input_parquet(&path);

    let aggregated_schema = Arc::new(Schema::new(vec![Field::new("avg_amount", DataType::Decimal128(19, 6), true)]));
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["amount".to_string()],
    );
    let aggregate = AggregateSpec::new(
        Vec::new(),
        vec![AggregateExpr::new(AggregateFunction::Avg, Expr::column("amount"), "avg_amount")],
    );

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let aggregate = builder.add_aggregate(source, aggregate, aggregated_schema);
    let plan = builder.finish_with_host_sink(aggregate);

    let context = ExecutionContext::builder()
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
    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.source_chunks_planned, 3);
    assert_eq!(metrics.source_chunks_committed, 3);
    assert_eq!(metrics.partial_aggregate_count, 3);
    assert_eq!(metrics.final_merge_attempts, 1);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 1);
    let avg_amount = decimal128_array(batch, 0);
    assert_eq!(avg_amount.value(0), 1_666_666);
    assert_eq!(avg_amount.precision(), 19);
    assert_eq!(avg_amount.scale(), 6);

    fs::remove_dir_all(dir).unwrap();
}
