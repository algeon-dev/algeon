#[test]
fn grant_backed_budget_tokens_share_arena_across_stream_leases() {
    let runtime = Arc::new(cudf::Runtime::builder().with_device_memory_manager().build().unwrap());
    let governor = MemoryGovernor::new(runtime.device().ordinal(), 64 * 1024 * 1024).unwrap();
    let grant =
        governor.request(GrantRequest::new(AdmissionRequestId::new(1), 12 * 1024 * 1024, 12 * 1024 * 1024)).unwrap();
    let context =
        ExecutionContext::builder().with_shared_runtime(Arc::clone(&runtime)).with_memory_grant(grant).build().unwrap();

    assert_eq!(governor.report().unwrap().outstanding_bytes, 12 * 1024 * 1024);

    let grant = context.memory_grant().unwrap().unwrap();
    let probe_token = context
        .execution_budget_token(cudf::ExecutionBudgetRequest::new(1024 * 1024).with_boundary("grant_arena_probe"))
        .unwrap();
    assert_eq!(
        probe_token
            .reservation_arena_usage()
            .unwrap()
            .expect("grant-backed token should expose arena usage")
            .allocation_limit_bytes,
        governor.report().unwrap().outstanding_bytes
    );
    assert!(grant.try_grow(4 * 1024 * 1024).unwrap());
    assert_eq!(governor.report().unwrap().outstanding_bytes, 16 * 1024 * 1024);
    assert_eq!(
        probe_token
            .reservation_arena_usage()
            .unwrap()
            .expect("grant-backed token should expose arena usage")
            .allocation_limit_bytes,
        16 * 1024 * 1024
    );
    assert!(grant.try_shrink_to(12 * 1024 * 1024).unwrap());
    assert_eq!(governor.report().unwrap().outstanding_bytes, 12 * 1024 * 1024);
    assert_eq!(
        probe_token
            .reservation_arena_usage()
            .unwrap()
            .expect("grant-backed token should expose arena usage")
            .allocation_limit_bytes,
        12 * 1024 * 1024
    );
    drop(probe_token);

    let left_token = context
        .execution_budget_token(cudf::ExecutionBudgetRequest::new(16 * 1024 * 1024).with_boundary("grant_arena_left"))
        .unwrap();
    assert_eq!(left_token.enforcement(), cudf::ExecutionBudgetEnforcement::ReservationArenaMember);
    let left_buffer = context.runtime().unwrap().device_buffer(8 * 1024 * 1024).unwrap();

    let stream_lease = context.service_stream_pool().acquire().unwrap();
    let leased_context = context.with_owned_stream_lease(stream_lease).unwrap();
    let right_context = leased_context.context();
    let right_token = right_context
        .execution_budget_token(cudf::ExecutionBudgetRequest::new(16 * 1024 * 1024).with_boundary("grant_arena_right"))
        .unwrap();
    assert_eq!(right_token.enforcement(), cudf::ExecutionBudgetEnforcement::ReservationArenaMember);

    let err = expect_device_buffer_error(right_context.runtime().unwrap().device_buffer(8 * 1024 * 1024));
    assert_eq!(err.kind(), cudf::ErrorKind::ResourceExhausted);

    drop(left_buffer);

    let right_buffer = right_context.runtime().unwrap().device_buffer(8 * 1024 * 1024).unwrap();
    assert_eq!(right_buffer.size_bytes().unwrap(), 8 * 1024 * 1024);
}

#[test]
fn shared_memory_grant_contexts_reuse_one_arena_parent() {
    let left_runtime = Arc::new(cudf::Runtime::builder().with_device_memory_manager().build().unwrap());
    let right_runtime = Arc::new(cudf::Runtime::builder().with_device_memory_manager().build().unwrap());
    let governor = MemoryGovernor::new(left_runtime.device().ordinal(), 64 * 1024 * 1024).unwrap();
    let grant =
        governor.request(GrantRequest::new(AdmissionRequestId::new(2), 12 * 1024 * 1024, 12 * 1024 * 1024)).unwrap();
    let left_context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&left_runtime))
        .with_memory_grant(grant.clone())
        .build()
        .unwrap();
    let right_context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&right_runtime))
        .with_memory_grant(grant)
        .build()
        .unwrap();

    let left_token = left_context
        .execution_budget_token(cudf::ExecutionBudgetRequest::new(16 * 1024 * 1024).with_boundary("shared_grant_left"))
        .unwrap();
    let right_token = right_context
        .execution_budget_token(cudf::ExecutionBudgetRequest::new(16 * 1024 * 1024).with_boundary("shared_grant_right"))
        .unwrap();
    assert_eq!(
        left_token.reservation_arena_usage().unwrap().unwrap().allocation_limit_bytes,
        right_token.reservation_arena_usage().unwrap().unwrap().allocation_limit_bytes
    );

    let left_buffer = left_context.runtime().unwrap().device_buffer(8 * 1024 * 1024).unwrap();
    let err = expect_device_buffer_error(right_context.runtime().unwrap().device_buffer(8 * 1024 * 1024));
    assert_eq!(err.kind(), cudf::ErrorKind::ResourceExhausted);
    drop(left_buffer);
}

#[test]
fn native_query_executes_direct_parquet_pipeline_to_host_sink() {
    let dir = unique_temp_dir("nexus-query-engine-native-execution");
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

    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime)).unwrap();
    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.parquet_files, 1);
    assert_eq!(metrics.read_columns, 4);
    assert_eq!(metrics.input_batches, 1);
    assert_eq!(metrics.input_rows, 4);
    assert_eq!(metrics.source_read_runtime_rows.len(), 1);
    let source_read_row = &metrics.source_read_runtime_rows[0];
    assert_eq!(source_read_row.source_node_id, source.to_string());
    assert_eq!(source_read_row.read_kind, "file");
    assert_eq!(source_read_row.read_strategy, "local_kvikio");
    assert_eq!(source_read_row.parquet_read_evidence.input_kind, "local_path");
    assert_eq!(source_read_row.parquet_read_evidence.remote_read_mode, "none");
    assert_eq!(source_read_row.parquet_read_evidence.backend, "libcudf_path_reader");
    assert!(source_read_row.parquet_read_evidence.pruning_input_row_groups.is_some());
    assert_eq!(source_read_row.parquet_read_evidence.row_groups_after_stats_filter, None);
    assert_eq!(source_read_row.parquet_read_evidence.row_groups_after_bloom_filter, None);
    assert_eq!(source_read_row.parquet_read_evidence.bloom_pruning_status, "no_reader_predicate");
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
    let dir = unique_temp_dir("nexus-query-engine-native-boundary-policy-source-host");
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
fn native_query_executes_multi_file_direct_parquet_source_in_file_order() {
    let dir = unique_temp_dir("nexus-query-engine-native-execution-multi-file-order");
    let first_path = dir.join("part-000.parquet");
    let second_path = dir.join("part-001.parquet");
    let third_path = dir.join("part-002.parquet");
    let schema = write_i32_parquet_with_row_group_rows(&first_path, &[10, 11], 2);
    write_i32_parquet_with_row_group_rows(&second_path, &[20, 21], 2);
    write_i32_parquet_with_row_group_rows(&third_path, &[30], 1);

    let source = LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new(&second_path).with_file_size_bytes(fs::metadata(&second_path).unwrap().len()),
            LocalParquetFile::new(&first_path).with_file_size_bytes(fs::metadata(&first_path).unwrap().len()),
            LocalParquetFile::new(&third_path).with_file_size_bytes(fs::metadata(&third_path).unwrap().len()),
        ],
        Arc::clone(&schema),
        vec!["id".to_string()],
    );
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let plan = builder.finish_with_host_sink(source);

    let context = ExecutionContext::from_shared_runtime(Arc::new(cudf::Runtime::new().unwrap())).unwrap();
    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.parquet_files, 3);
    assert_eq!(metrics.input_rows, 5);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 5);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[20, 21, 10, 11, 30]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_query_executes_multi_file_direct_parquet_source() {
    let dir = unique_temp_dir("nexus-query-engine-native-execution-multi-file");
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

    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime)).unwrap();
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
