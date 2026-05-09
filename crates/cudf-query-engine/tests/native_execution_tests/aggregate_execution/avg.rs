use super::*;

#[test]
fn bounded_native_query_executes_avg_from_sum_count_state() {
    let dir = unique_temp_dir("cudf-query-engine-bounded-native-avg");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_uneven_avg_input_parquet(&path);

    let aggregated_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("avg_m00", DataType::Float64, true),
        Field::new("count_m00", DataType::Int64, false),
    ]));

    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["key".to_string(), "payload".to_string(), "m00".to_string()],
    );
    let filter = Expr::binary(Expr::column("payload"), BinaryOp::GtEq, Expr::literal(LiteralValue::Int64(10)));
    let aggregate = AggregateSpec::new(
        vec![Expr::column("key")],
        vec![
            AggregateExpr::new(AggregateFunction::Avg, Expr::column("m00"), "avg_m00"),
            AggregateExpr::new(AggregateFunction::Count, Expr::column("m00"), "count_m00"),
        ],
    );
    let sort = SortSpec::new(vec![
        SortExpr::new(Expr::column("avg_m00"), true, false),
        SortExpr::new(Expr::column("key"), false, false),
    ]);

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let filter = builder.add_filter(source, filter, Arc::clone(&input_schema));
    let aggregate = builder.add_aggregate(filter, aggregate, Arc::clone(&aggregated_schema));
    let sort = builder.add_sort(aggregate, sort, Arc::clone(&aggregated_schema));
    let limit = builder.add_limit(sort, LimitSpec::fetch(3), aggregated_schema);
    let plan = builder.finish_with_host_sink(limit);

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
    assert_budgeted_success_event_with_rows(metrics, "cudf_query_engine.aggregate.finalize_avg", 3);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 3);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[3, 2, 1]);
    assert_eq!(float64_array(batch, 1).values().as_ref(), &[300.0, 150.0, 20.0]);
    assert_eq!(int64_array(batch, 2).values().as_ref(), &[1, 2, 3]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_executes_single_chunk_avg_finalization() {
    let dir = unique_temp_dir("cudf-query-engine-bounded-native-single-chunk-avg");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_uneven_avg_input_parquet(&path);

    let aggregated_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("avg_m00", DataType::Float64, true),
    ]));

    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["key".to_string(), "payload".to_string(), "m00".to_string()],
    );
    let aggregate = AggregateSpec::new(
        vec![Expr::column("key")],
        vec![AggregateExpr::new(AggregateFunction::Avg, Expr::column("m00"), "avg_m00")],
    );
    let sort = SortSpec::new(vec![SortExpr::new(Expr::column("key"), false, false)]);

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let aggregate = builder.add_aggregate(source, aggregate, Arc::clone(&aggregated_schema));
    let sort = builder.add_sort(aggregate, sort, Arc::clone(&aggregated_schema));
    let limit = builder.add_limit(sort, LimitSpec::fetch(3), aggregated_schema);
    let plan = builder.finish_with_host_sink(limit);

    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 8 * 1024 * 1024 * 1024,
            max_source_chunk_bytes: 1024 * 1024,
            max_row_groups_per_chunk: 8,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .build()
        .unwrap();
    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.source_chunks_planned, 1);
    assert_eq!(metrics.final_merge_attempts, 0);
    assert_budgeted_success_event_with_rows(metrics, "cudf_query_engine.aggregate.finalize_avg", 3);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[1, 2, 3]);
    assert_eq!(float64_array(batch, 1).values().as_ref(), &[20.0, 150.0, 300.0]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_query_executes_grouped_avg_with_float32_output_schema() {
    let dir = unique_temp_dir("cudf-query-engine-native-grouped-avg-float32");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_uneven_avg_input_parquet(&path);

    let aggregated_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("avg_m00", DataType::Float32, true),
    ]));

    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["key".to_string(), "m00".to_string()],
    );
    let aggregate = AggregateSpec::new(
        vec![Expr::column("key")],
        vec![AggregateExpr::new(AggregateFunction::Avg, Expr::column("m00"), "avg_m00")],
    );
    let sort = SortSpec::new(vec![SortExpr::new(Expr::column("key"), false, false)]);

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let aggregate = builder.add_aggregate(source, aggregate, Arc::clone(&aggregated_schema));
    let sort = builder.add_sort(aggregate, sort, Arc::clone(&aggregated_schema));
    let plan = builder.finish_with_host_sink(sort);

    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime));
    let result = execute_query_plan(&context, &plan).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.schema(), aggregated_schema);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[1, 2, 3]);
    assert_eq!(float32_array(batch, 1).values().as_ref(), &[20.0, 150.0, 300.0]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_executes_nullable_avg_with_null_all_group() {
    let dir = unique_temp_dir("cudf-query-engine-bounded-native-nullable-avg");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_nullable_avg_input_parquet(&path);

    let aggregated_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("avg_m00", DataType::Float64, true),
    ]));

    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["key".to_string(), "m00".to_string()],
    );
    let aggregate = AggregateSpec::new(
        vec![Expr::column("key")],
        vec![AggregateExpr::new(AggregateFunction::Avg, Expr::column("m00"), "avg_m00")],
    );
    let sort = SortSpec::new(vec![SortExpr::new(Expr::column("key"), false, false)]);

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let aggregate = builder.add_aggregate(source, aggregate, Arc::clone(&aggregated_schema));
    let sort = builder.add_sort(aggregate, sort, Arc::clone(&aggregated_schema));
    let limit = builder.add_limit(sort, LimitSpec::fetch(3), aggregated_schema);
    let plan = builder.finish_with_host_sink(limit);

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

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    let avg = float64_array(batch, 1);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[1, 2, 3]);
    assert!(avg.is_null(0));
    assert_eq!(avg.value(1), 10.0);
    assert_eq!(avg.value(2), 30.0);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_rejects_avg_with_unsupported_output_schema() {
    let dir = unique_temp_dir("cudf-query-engine-bounded-native-avg-output-type");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_uneven_avg_input_parquet(&path);

    let aggregated_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("avg_m00", DataType::Int64, true),
    ]));

    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["key".to_string(), "m00".to_string()],
    );
    let aggregate = AggregateSpec::new(
        vec![Expr::column("key")],
        vec![AggregateExpr::new(AggregateFunction::Avg, Expr::column("m00"), "avg_m00")],
    );
    let sort = SortSpec::new(vec![SortExpr::new(Expr::column("key"), false, false)]);

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let aggregate = builder.add_aggregate(source, aggregate, Arc::clone(&aggregated_schema));
    let sort = builder.add_sort(aggregate, sort, Arc::clone(&aggregated_schema));
    let limit = builder.add_limit(sort, LimitSpec::fetch(3), aggregated_schema);
    let plan = builder.finish_with_host_sink(limit);

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

    match execute_query_plan(&context, &plan) {
        Ok(_) => panic!("avg with unsupported output schema should be rejected"),
        Err(error) => {
            assert_eq!(error.code(), QueryErrorCode::UnsupportedType);
            assert_eq!(error.operation(), "cudf_query_engine.aggregate.validate_avg");
            assert_eq!(error.fact_value("alias"), Some("avg_m00"));
        },
    }

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_query_rejects_grouped_avg_with_unsupported_output_schema_before_source_read() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("m00", DataType::Int64, false),
    ]));
    let aggregated_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("avg_m00", DataType::Int64, true),
    ]));

    let source = LocalParquetSource::new(
        "/tmp/does-not-need-to-exist-for-avg-output-validation.parquet",
        Arc::clone(&input_schema),
        vec!["key".to_string(), "m00".to_string()],
    );
    let aggregate = AggregateSpec::new(
        vec![Expr::column("key")],
        vec![AggregateExpr::new(AggregateFunction::Avg, Expr::column("m00"), "avg_m00")],
    );

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let aggregate = builder.add_aggregate(source, aggregate, aggregated_schema);
    let plan = builder.finish_with_host_sink(aggregate);
    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime));

    match execute_query_plan(&context, &plan) {
        Ok(_) => panic!("grouped avg with unsupported output schema should be rejected before source read"),
        Err(error) => {
            assert_eq!(error.code(), QueryErrorCode::UnsupportedType);
            assert_eq!(error.operation(), "cudf_query_engine.aggregate.validate_avg");
            assert_eq!(error.fact_value("alias"), Some("avg_m00"));
            assert_eq!(error.fact_value("data_type"), Some("Int64"));
        },
    }
}

#[test]
fn native_query_rejects_global_avg_reserved_state_alias_collision_before_source_read() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = Arc::new(Schema::new(vec![Field::new("m00", DataType::Int64, false)]));
    let aggregated_schema = Arc::new(Schema::new(vec![Field::new("__cudf_avg_0_sum", DataType::Float64, true)]));
    let aggregate = AggregateSpec::new(
        Vec::new(),
        vec![AggregateExpr::new(AggregateFunction::Avg, Expr::column("m00"), "__cudf_avg_0_sum")],
    );

    let mut builder = QueryPlanBuilder::new();
    let source =
        builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("missing_input", Arc::clone(&input_schema))));
    let aggregate = builder.add_aggregate(source, aggregate, aggregated_schema);
    let plan = builder.finish_with_host_sink(aggregate);
    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime));

    match execute_query_plan(&context, &plan) {
        Ok(_) => panic!("global avg reserved state alias collision should be rejected before source read"),
        Err(error) => {
            assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
            assert_eq!(error.operation(), "cudf_query_engine.global_aggregate.validate_avg_state_alias");
            assert_eq!(error.fact_value("alias"), Some("__cudf_avg_0_sum"));
            assert_eq!(error.fact_value("reserved_aliases"), Some("__cudf_avg_0_sum,__cudf_avg_0_count"));
        },
    }
}
