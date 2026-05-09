use super::*;

#[test]
fn native_query_executes_direct_parquet_pipeline_to_host_sink() {
    let dir = unique_temp_dir("cudf-query-engine-native-execution");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_input_parquet(&runtime, &path);

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
                .with_row_group_bytes(vec![4096, 4096, 4096]),
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
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let filter = builder.add_filter(source, filter, Arc::clone(&input_schema));
    let projection = builder.add_projection(filter, projection_outputs(projection, projected_schema));
    let aggregate = builder.add_aggregate(projection, aggregate, Arc::clone(&aggregated_schema));
    let sort = builder.add_sort(aggregate, sort, Arc::clone(&aggregated_schema));
    let limit = builder.add_limit(sort, LimitSpec::fetch(2), aggregated_schema);
    let plan = builder.finish_with_host_sink(limit);

    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime));
    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.parquet_files, 1);
    assert_eq!(metrics.read_columns, 4);
    assert_eq!(metrics.input_batches, 1);
    assert_eq!(metrics.input_rows, 4);
    assert_eq!(metrics.boundary.host_to_device_uploads, 0);
    assert_eq!(metrics.boundary.device_to_host_materializations, 1);
    assert_eq!(metrics.boundary.device_to_host_materialized_rows, 2);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 2);
    assert_eq!(batch.num_columns(), 2);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[3, 1]);
    assert_eq!(int64_array(batch, 1).values().as_ref(), &[404, 303]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_execution_boundary_policy_reports_direct_source_and_explicit_host_materialization() {
    let dir = unique_temp_dir("cudf-query-engine-native-boundary-policy-source-host");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_input_parquet(&runtime, &path);
    let policy = ExecutionBoundaryPolicy::default();

    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["key".to_string(), "payload".to_string(), "m00".to_string(), "m01".to_string()],
    );

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let plan = builder.finish_with_host_sink(source);
    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_boundary_policy(policy)
        .build()
        .unwrap();

    assert_eq!(context.boundary_policy(), policy);
    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.execution_boundary_policy, policy);
    assert_eq!(metrics.boundary.host_to_device_uploads, 0);
    assert_eq!(metrics.boundary.device_to_host_materializations, 1);
    assert_eq!(metrics.boundary.device_to_host_materialized_rows, 4);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("explicit host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 4);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_query_executes_group_only_aggregate_as_distinct_keys() {
    let dir = unique_temp_dir("cudf-query-engine-native-group-only-aggregate");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_input_parquet(&runtime, &path);
    let output_schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, false)]));

    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["key".to_string()],
    );
    let aggregate = AggregateSpec::new(vec![Expr::column("key")], vec![]);
    let sort = SortSpec::new(vec![SortExpr::new(Expr::column("key"), false, false)]);

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let aggregate = builder.add_aggregate(source, aggregate, Arc::clone(&output_schema));
    let sort = builder.add_sort(aggregate, sort, Arc::clone(&output_schema));
    let plan = builder.finish_with_host_sink(sort);

    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime));
    let result = execute_query_plan(&context, &plan).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 3);
    assert_eq!(batch.num_columns(), 1);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[1, 2, 3]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_executes_group_only_aggregate_with_budgeted_distinct() {
    let dir = unique_temp_dir("cudf-query-engine-bounded-group-only-aggregate");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_final_merge_retry_input_parquet(&path, 4);
    let output_schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, false)]));

    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["key".to_string()],
    );
    let aggregate = AggregateSpec::new(vec![Expr::column("key")], vec![]);
    let sort = SortSpec::new(vec![SortExpr::new(Expr::column("key"), false, false)]);

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let aggregate = builder.add_aggregate(source, aggregate, Arc::clone(&output_schema));
    let sort = builder.add_sort(aggregate, sort, Arc::clone(&output_schema));
    let plan = builder.finish_with_host_sink(sort);

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
    assert_eq!(metrics.source_chunks_committed, 3);
    assert_eq!(metrics.final_merge_attempts, 1);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 4);
    assert_eq!(batch.num_columns(), 1);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[0, 1, 2, 3]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_query_executes_multi_file_direct_parquet_source() {
    let dir = unique_temp_dir("cudf-query-engine-native-execution-multi-file");
    let first_path = dir.join("part-000.parquet");
    let second_path = dir.join("part-001.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_input_parquet(&runtime, &first_path);
    write_input_parquet(&runtime, &second_path);

    let projected_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("x00", DataType::Int64, false),
    ]));
    let aggregated_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("sum_x00", DataType::Int64, true),
    ]));
    let files = vec![
        LocalParquetFile::new(&first_path).with_file_size_bytes(fs::metadata(&first_path).unwrap().len()),
        LocalParquetFile::new(&second_path).with_file_size_bytes(fs::metadata(&second_path).unwrap().len()),
    ];
    let source = LocalParquetSource::from_files(
        files,
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
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let filter = builder.add_filter(source, filter, Arc::clone(&input_schema));
    let projection = builder.add_projection(filter, projection_outputs(projection, projected_schema));
    let aggregate = builder.add_aggregate(projection, aggregate, Arc::clone(&aggregated_schema));
    let sort = builder.add_sort(aggregate, sort, Arc::clone(&aggregated_schema));
    let limit = builder.add_limit(sort, LimitSpec::fetch(2), aggregated_schema);
    let plan = builder.finish_with_host_sink(limit);

    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime));
    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.parquet_files, 2);
    assert_eq!(metrics.read_columns, 8);
    assert_eq!(metrics.input_batches, 2);
    assert_eq!(metrics.input_rows, 8);
    assert_eq!(metrics.boundary.host_to_device_uploads, 0);
    assert_eq!(metrics.boundary.device_to_host_materializations, 1);
    assert_eq!(metrics.boundary.device_to_host_materialized_rows, 2);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[3, 1]);
    assert_eq!(int64_array(batch, 1).values().as_ref(), &[808, 606]);

    fs::remove_dir_all(dir).unwrap();
}
