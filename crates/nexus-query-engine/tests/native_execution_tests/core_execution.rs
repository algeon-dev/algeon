use super::*;

fn expect_device_buffer_error(result: cudf::Result<cudf::core::DeviceBuffer>) -> cudf::Error {
    match result {
        Ok(_) => panic!("expected device buffer allocation to fail"),
        Err(error) => error,
    }
}

#[test]
fn grant_backed_budget_tokens_share_arena_across_stream_leases() {
    let runtime = Arc::new(cudf::Runtime::builder().with_device_memory_manager().build().unwrap());
    let governor = MemoryGovernor::new(runtime.device().ordinal(), 64 * 1024 * 1024).unwrap();
    let grant = governor.request(GrantRequest::new(1, 12 * 1024 * 1024, 12 * 1024 * 1024)).unwrap();
    let context =
        ExecutionContext::builder().with_shared_runtime(Arc::clone(&runtime)).with_memory_grant(grant).build().unwrap();

    assert_eq!(governor.report().unwrap().outstanding_bytes, 12 * 1024 * 1024);

    let grant = context.memory_grant().unwrap().clone();
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
    let left_buffer = context.runtime().device_buffer(8 * 1024 * 1024).unwrap();

    let stream_lease = context.service_stream_pool().acquire().unwrap();
    let leased_context = context.with_owned_stream_lease(stream_lease).unwrap();
    let right_context = leased_context.context();
    let right_token = right_context
        .execution_budget_token(cudf::ExecutionBudgetRequest::new(16 * 1024 * 1024).with_boundary("grant_arena_right"))
        .unwrap();
    assert_eq!(right_token.enforcement(), cudf::ExecutionBudgetEnforcement::ReservationArenaMember);

    let err = expect_device_buffer_error(right_context.runtime().device_buffer(8 * 1024 * 1024));
    assert_eq!(err.kind(), cudf::ErrorKind::ResourceExhausted);

    drop(left_buffer);

    let right_buffer = right_context.runtime().device_buffer(8 * 1024 * 1024).unwrap();
    assert_eq!(right_buffer.size_bytes().unwrap(), 8 * 1024 * 1024);
}

#[test]
fn shared_memory_grant_contexts_reuse_one_arena_parent() {
    let left_runtime = Arc::new(cudf::Runtime::builder().with_device_memory_manager().build().unwrap());
    let right_runtime = Arc::new(cudf::Runtime::builder().with_device_memory_manager().build().unwrap());
    let governor = MemoryGovernor::new(left_runtime.device().ordinal(), 64 * 1024 * 1024).unwrap();
    let grant = governor.request(GrantRequest::new(2, 12 * 1024 * 1024, 12 * 1024 * 1024)).unwrap();
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

    let left_buffer = left_context.runtime().device_buffer(8 * 1024 * 1024).unwrap();
    let err = expect_device_buffer_error(right_context.runtime().device_buffer(8 * 1024 * 1024));
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
    assert_eq!(source_read_row.parquet_read_evidence.fallback_reason, "not_applicable");
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
fn native_local_parquet_source_applies_iceberg_position_deletes_before_output_and_count_probe() {
    let dir = unique_temp_dir("nexus-query-engine-native-position-deletes");
    let data_path = dir.join("data.parquet");
    let delete_path = dir.join("delete.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    write_input_parquet(&runtime, &data_path);
    write_position_delete_parquet(&delete_path, data_path.display().to_string().as_str(), &[1, 3], &[0]);

    let delete_file = ParquetPositionDeleteFile::from_resolved_parquet_source(
        "delete-0001",
        ResolvedParquetSource::local_kvikio(delete_path.display().to_string()),
    )
    .unwrap()
    .with_file_size_bytes(fs::metadata(&delete_path).unwrap().len());
    let data_file = LocalParquetFile::new(&data_path)
        .with_file_size_bytes(fs::metadata(&data_path).unwrap().len())
        .with_position_delete_files(vec![delete_file]);

    let projected_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int64, false),
    ]));
    let source = LocalParquetSource::from_files(
        vec![data_file.clone()],
        Arc::clone(&projected_schema),
        vec!["key".to_string(), "payload".to_string()],
    );
    let mut builder = QueryPlanBuilder::new();
    let source_node = builder.add_source(SourceNode::LocalParquet(source));
    let plan = builder.finish_with_host_sink(source_node);
    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime)).unwrap();

    let result = execute_query_plan(&context, &plan).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 2);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[1, 1]);
    assert_eq!(int64_array(batch, 1).values().as_ref(), &[5, 20]);

    let limited_source = LocalParquetSource::from_files(
        vec![data_file.clone()],
        projected_schema,
        vec!["key".to_string(), "payload".to_string()],
    )
    .with_scan_limit(Some(1));
    let mut builder = QueryPlanBuilder::new();
    let source_node = builder.add_source(SourceNode::LocalParquet(limited_source));
    let plan = builder.finish_with_host_sink(source_node);

    let result = execute_query_plan(&context, &plan).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize scan-limited RecordBatch");
    };
    assert_eq!(batch.num_rows(), 1);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[1]);
    assert_eq!(int64_array(batch, 1).values().as_ref(), &[5]);

    let carrier_schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, false)]));
    let count_source = LocalParquetSource::count_star_probe(vec![data_file], carrier_schema, vec!["key".to_string()])
        .with_column_roles(Vec::new(), vec!["key".to_string()]);
    let mut builder = QueryPlanBuilder::new();
    let source_node = builder.add_source(SourceNode::LocalParquet(count_source));
    let plan = builder.finish_with_host_sink(source_node);

    let result = execute_query_plan(&context, &plan).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize count probe RecordBatch");
    };
    assert_eq!(batch.num_rows(), 1);
    assert_eq!(int64_array(batch, 0).values().as_ref(), &[2]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_local_parquet_source_applies_iceberg_equality_deletes_before_output_limit_and_count_probe() {
    let dir = unique_temp_dir("nexus-query-engine-native-equality-deletes");
    let data_path = dir.join("data.parquet");
    let delete_path = dir.join("delete.parquet");
    let limited_data_path = dir.join("limited-data.parquet");
    let limited_delete_path = dir.join("limited-delete.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime)).unwrap();
    let data_schema = write_nullable_key_payload_parquet(&runtime, &data_path);
    write_equality_delete_key_parquet(&runtime, &delete_path, &[Some(1), None]);

    let delete_file = ParquetEqualityDeleteFile::from_resolved_parquet_source(
        "delete-0001",
        ResolvedParquetSource::local_kvikio(delete_path.display().to_string()),
        vec![1],
        vec!["key".to_string()],
    )
    .unwrap()
    .with_file_size_bytes(fs::metadata(&delete_path).unwrap().len())
    .with_row_count(2);
    let data_file = LocalParquetFile::new(&data_path)
        .with_file_size_bytes(fs::metadata(&data_path).unwrap().len())
        .with_equality_delete_files(vec![delete_file]);

    let source = LocalParquetSource::from_files(
        vec![data_file.clone()],
        Arc::clone(&data_schema),
        vec!["key".to_string(), "payload".to_string()],
    )
    .with_column_roles(vec!["payload".to_string()], vec!["key".to_string()]);
    let output_schema = Arc::new(Schema::new(vec![Field::new("payload", DataType::Int64, false)]));
    let mut builder = QueryPlanBuilder::new();
    let source_node = builder.add_source(SourceNode::LocalParquet(source));
    let projection = builder
        .add_projection(source_node, projection_outputs(vec![Expr::column("payload")], Arc::clone(&output_schema)));
    let plan = builder.finish_with_host_sink(projection);

    let result = execute_query_plan(&context, &plan).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize equality-delete RecordBatch");
    };
    assert_eq!(batch.num_rows(), 2);
    assert_eq!(int64_array(batch, 0).values().as_ref(), &[10, 40]);

    let limited_data_schema = write_key_payload_row_grouped_parquet(&limited_data_path, 1);
    write_equality_delete_key_parquet(&runtime, &limited_delete_path, &[Some(1)]);
    let limited_delete_file = ParquetEqualityDeleteFile::from_resolved_parquet_source(
        "delete-0002",
        ResolvedParquetSource::local_kvikio(limited_delete_path.display().to_string()),
        vec![1],
        vec!["key".to_string()],
    )
    .unwrap()
    .with_file_size_bytes(fs::metadata(&limited_delete_path).unwrap().len())
    .with_row_count(1);
    let limited_data_file = LocalParquetFile::new(&limited_data_path)
        .with_file_size_bytes(fs::metadata(&limited_data_path).unwrap().len())
        .with_equality_delete_files(vec![limited_delete_file]);
    let limited_source = LocalParquetSource::from_files(
        vec![limited_data_file],
        limited_data_schema,
        vec!["key".to_string(), "payload".to_string()],
    )
    .with_column_roles(vec!["payload".to_string()], vec!["key".to_string()])
    .with_scan_limit(Some(1));
    let mut builder = QueryPlanBuilder::new();
    let source_node = builder.add_source(SourceNode::LocalParquet(limited_source));
    let projection = builder
        .add_projection(source_node, projection_outputs(vec![Expr::column("payload")], Arc::clone(&output_schema)));
    let plan = builder.finish_with_host_sink(projection);

    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize scan-limited equality-delete RecordBatch");
    };
    assert_eq!(batch.num_rows(), 1);
    assert_eq!(int64_array(batch, 0).values().as_ref(), &[10]);
    let chunk_reads =
        metrics.source_read_runtime_rows.iter().filter(|row| row.read_kind == "chunk").collect::<Vec<_>>();
    assert_eq!(chunk_reads.len(), 2);
    assert!(chunk_reads.iter().all(|row| row.row_group_count == 1));

    let carrier_schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, true)]));
    let count_source = LocalParquetSource::count_star_probe(vec![data_file], carrier_schema, vec!["key".to_string()])
        .with_column_roles(Vec::new(), vec!["key".to_string()]);
    let mut builder = QueryPlanBuilder::new();
    let source_node = builder.add_source(SourceNode::LocalParquet(count_source));
    let plan = builder.finish_with_host_sink(source_node);

    let result = execute_query_plan(&context, &plan).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize equality-delete count probe RecordBatch");
    };
    assert_eq!(batch.num_rows(), 1);
    assert_eq!(int64_array(batch, 0).values().as_ref(), &[2]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_local_parquet_source_coerces_widened_equality_delete_keys_before_anti_join() {
    let dir = unique_temp_dir("nexus-query-engine-native-widened-equality-deletes");
    let data_path = dir.join("data.parquet");
    let delete_path = dir.join("delete.parquet");
    let limited_data_path = dir.join("limited-data.parquet");
    let limited_delete_path = dir.join("limited-delete.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime)).unwrap();
    write_nullable_key_payload_parquet(&runtime, &data_path);
    write_equality_delete_i64_key_parquet(&runtime, &delete_path, &[Some(1), None]);

    let contract_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int64, true),
        Field::new("payload", DataType::Int64, false),
    ]));
    let delete_file = ParquetEqualityDeleteFile::from_resolved_parquet_source(
        "delete-0001",
        ResolvedParquetSource::local_kvikio(delete_path.display().to_string()),
        vec![1],
        vec!["key".to_string()],
    )
    .unwrap()
    .with_file_size_bytes(fs::metadata(&delete_path).unwrap().len())
    .with_row_count(2);
    let data_file = LocalParquetFile::new(&data_path)
        .with_file_size_bytes(fs::metadata(&data_path).unwrap().len())
        .with_equality_delete_files(vec![delete_file]);

    let source = LocalParquetSource::from_files(
        vec![data_file.clone()],
        Arc::clone(&contract_schema),
        vec!["key".to_string(), "payload".to_string()],
    )
    .with_column_roles(vec!["payload".to_string()], vec!["key".to_string()]);
    let output_schema = Arc::new(Schema::new(vec![Field::new("payload", DataType::Int64, false)]));
    let mut builder = QueryPlanBuilder::new();
    let source_node = builder.add_source(SourceNode::LocalParquet(source));
    let projection = builder
        .add_projection(source_node, projection_outputs(vec![Expr::column("payload")], Arc::clone(&output_schema)));
    let plan = builder.finish_with_host_sink(projection);

    let result = execute_query_plan(&context, &plan).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize widened equality-delete RecordBatch");
    };
    assert_eq!(batch.num_rows(), 2);
    assert_eq!(int64_array(batch, 0).values().as_ref(), &[10, 40]);

    let limited_contract_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int64, false),
        Field::new("payload", DataType::Int64, false),
    ]));
    write_key_payload_row_grouped_parquet(&limited_data_path, 1);
    write_equality_delete_i64_key_parquet(&runtime, &limited_delete_path, &[Some(1)]);
    let limited_delete_file = ParquetEqualityDeleteFile::from_resolved_parquet_source(
        "delete-0002",
        ResolvedParquetSource::local_kvikio(limited_delete_path.display().to_string()),
        vec![1],
        vec!["key".to_string()],
    )
    .unwrap()
    .with_file_size_bytes(fs::metadata(&limited_delete_path).unwrap().len())
    .with_row_count(1);
    let limited_data_file = LocalParquetFile::new(&limited_data_path)
        .with_file_size_bytes(fs::metadata(&limited_data_path).unwrap().len())
        .with_equality_delete_files(vec![limited_delete_file]);
    let limited_source = LocalParquetSource::from_files(
        vec![limited_data_file],
        limited_contract_schema,
        vec!["key".to_string(), "payload".to_string()],
    )
    .with_column_roles(vec!["payload".to_string()], vec!["key".to_string()])
    .with_scan_limit(Some(1));
    let mut builder = QueryPlanBuilder::new();
    let source_node = builder.add_source(SourceNode::LocalParquet(limited_source));
    let projection = builder
        .add_projection(source_node, projection_outputs(vec![Expr::column("payload")], Arc::clone(&output_schema)));
    let plan = builder.finish_with_host_sink(projection);

    let result = execute_query_plan(&context, &plan).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize scan-limited widened equality-delete RecordBatch");
    };
    assert_eq!(batch.num_rows(), 1);
    assert_eq!(int64_array(batch, 0).values().as_ref(), &[10]);

    let carrier_schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int64, true)]));
    let count_source = LocalParquetSource::count_star_probe(vec![data_file], carrier_schema, vec!["key".to_string()])
        .with_column_roles(Vec::new(), vec!["key".to_string()]);
    let mut builder = QueryPlanBuilder::new();
    let source_node = builder.add_source(SourceNode::LocalParquet(count_source));
    let plan = builder.finish_with_host_sink(source_node);

    let result = execute_query_plan(&context, &plan).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize widened equality-delete count probe RecordBatch");
    };
    assert_eq!(batch.num_rows(), 1);
    assert_eq!(int64_array(batch, 0).values().as_ref(), &[2]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_scan_limited_position_delete_source_reads_until_visible_row() {
    let dir = unique_temp_dir("nexus-query-engine-native-position-delete-limit");
    let data_path = dir.join("data.parquet");
    let delete_first_path = dir.join("delete-first.parquet");
    let delete_wrong_reference_path = dir.join("delete-wrong-reference.parquet");
    let delete_late_path = dir.join("delete-late.parquet");
    let projected_schema = write_single_i32_row_grouped_parquet(&data_path, 1);
    write_position_delete_parquet(
        &delete_first_path,
        format!("file://localhost{}", data_path.display()).as_str(),
        &[0],
        &[],
    );
    write_position_delete_parquet(&delete_wrong_reference_path, data_path.display().to_string().as_str(), &[1], &[]);
    write_position_delete_parquet(&delete_late_path, data_path.display().to_string().as_str(), &[3], &[]);

    let delete_first = ParquetPositionDeleteFile::from_resolved_parquet_source(
        "delete-first",
        ResolvedParquetSource::local_kvikio(delete_first_path.display().to_string()),
    )
    .unwrap()
    .with_file_size_bytes(fs::metadata(&delete_first_path).unwrap().len())
    .with_row_count(1)
    .with_position_bounds(Some(0), Some(0));
    let delete_late = ParquetPositionDeleteFile::from_resolved_parquet_source(
        "delete-late",
        ResolvedParquetSource::local_kvikio(delete_late_path.display().to_string()),
    )
    .unwrap()
    .with_file_size_bytes(fs::metadata(&delete_late_path).unwrap().len())
    .with_row_count(1)
    .with_position_bounds(Some(3), Some(3));
    let delete_wrong_reference = ParquetPositionDeleteFile::from_resolved_parquet_source(
        "delete-wrong-reference",
        ResolvedParquetSource::local_kvikio(delete_wrong_reference_path.display().to_string()),
    )
    .unwrap()
    .with_file_size_bytes(fs::metadata(&delete_wrong_reference_path).unwrap().len())
    .with_row_count(1)
    .with_referenced_data_file(dir.join("other-data.parquet").display().to_string())
    .with_position_bounds(Some(1), Some(1));
    let data_file = LocalParquetFile::new(&data_path)
        .with_file_size_bytes(fs::metadata(&data_path).unwrap().len())
        .with_position_delete_files(vec![delete_first, delete_wrong_reference, delete_late]);
    let source = LocalParquetSource::from_files(vec![data_file], projected_schema, vec!["id".to_string()])
        .with_scan_limit(Some(1));
    let mut builder = QueryPlanBuilder::new();
    let source_node = builder.add_source(SourceNode::LocalParquet(source));
    let plan = builder.finish_with_host_sink(source_node);
    let context = ExecutionContext::from_shared_runtime(Arc::new(cudf::Runtime::new().unwrap())).unwrap();

    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize scan-limited RecordBatch");
    };
    assert_eq!(batch.num_rows(), 1);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[2]);

    let chunk_reads =
        metrics.source_read_runtime_rows.iter().filter(|row| row.read_kind == "chunk").collect::<Vec<_>>();
    assert_eq!(chunk_reads.len(), 2);
    assert!(chunk_reads.iter().all(|row| row.row_group_count == 1));
    assert_eq!(metrics.source_read_runtime_rows.iter().filter(|row| row.read_kind == "file").count(), 1);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_position_delete_reader_predicate_prunes_delete_row_groups_for_scan_limit() {
    let dir = unique_temp_dir("nexus-query-engine-native-position-delete-reader-predicate");
    let data_path = dir.join("data.parquet");
    let delete_path = dir.join("delete.parquet");
    let projected_schema = write_single_i32_row_grouped_parquet(&data_path, 1);
    write_position_delete_parquet_with_row_group_rows(
        &delete_path,
        data_path.display().to_string().as_str(),
        &[2, 3, 4, 5],
        &[],
        1,
    );

    let delete_file = ParquetPositionDeleteFile::from_resolved_parquet_source(
        "delete-broad",
        ResolvedParquetSource::local_kvikio(delete_path.display().to_string()),
    )
    .unwrap()
    .with_file_size_bytes(fs::metadata(&delete_path).unwrap().len())
    .with_row_count(4);
    let data_file = LocalParquetFile::new(&data_path)
        .with_file_size_bytes(fs::metadata(&data_path).unwrap().len())
        .with_position_delete_files(vec![delete_file]);
    let source = LocalParquetSource::from_files(vec![data_file], projected_schema, vec!["id".to_string()])
        .with_scan_limit(Some(1));
    let mut builder = QueryPlanBuilder::new();
    let source_node = builder.add_source(SourceNode::LocalParquet(source));
    let plan = builder.finish_with_host_sink(source_node);
    let context = ExecutionContext::from_shared_runtime(Arc::new(cudf::Runtime::new().unwrap())).unwrap();

    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize scan-limited RecordBatch");
    };
    assert_eq!(batch.num_rows(), 1);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[1]);

    let chunk_reads =
        metrics.source_read_runtime_rows.iter().filter(|row| row.read_kind == "chunk").collect::<Vec<_>>();
    assert_eq!(chunk_reads.len(), 1);
    let delete_reads = metrics
        .source_read_runtime_rows
        .iter()
        .filter(|row| row.read_kind == "file" && row.reader_predicate)
        .collect::<Vec<_>>();
    assert_eq!(delete_reads.len(), 1);
    let delete_read = delete_reads[0];
    assert_eq!(delete_read.output_rows, 0);
    let pruning_input = delete_read
        .parquet_read_evidence
        .pruning_input_row_groups
        .expect("position-delete reader predicate should report pruning input row groups");
    let after_stats = delete_read
        .parquet_read_evidence
        .row_groups_after_stats_filter
        .expect("position-delete reader predicate should report stats pruning");
    assert!(
        after_stats < pruning_input,
        "position-delete reader predicate should prune delete row groups: {delete_read:?}"
    );

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_local_parquet_source_carries_timestamp_timezone_to_host_output() {
    let dir = unique_temp_dir("nexus-query-engine-native-timestamp-tz");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("event_ts", DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())), true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(
                TimestampMicrosecondArray::from(vec![Some(1_700_000_000_000_000), None, Some(1_700_000_000_000_123)])
                    .with_timezone("UTC"),
            ),
        ],
    )
    .unwrap();
    let file = File::create(&path).unwrap();
    let mut writer =
        ArrowWriter::try_new(file, Arc::clone(&schema), Some(WriterProperties::builder().build())).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();

    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&schema),
        vec!["id".to_string(), "event_ts".to_string()],
    );
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let plan = builder.finish_with_host_sink(source);
    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime)).unwrap();

    let result = execute_query_plan(&context, &plan).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize timestamp-with-timezone output");
    };
    assert_eq!(batch.schema(), schema);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[1, 2, 3]);
    let event_ts = batch.column(1).as_primitive::<arrow::datatypes::TimestampMicrosecondType>();
    assert_eq!(event_ts.values().as_ref(), &[1_700_000_000_000_000, 0, 1_700_000_000_000_123]);
    assert!(event_ts.is_valid(0));
    assert!(!event_ts.is_valid(1));
    assert!(event_ts.is_valid(2));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_local_parquet_source_carries_list_utf8_through_scalar_filter_to_host_output() {
    let dir = unique_temp_dir("nexus-query-engine-native-list-utf8");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let source_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("tags", DataType::List(Arc::new(Field::new("element", DataType::Utf8, true))), true),
        Field::new("payload", DataType::Int64, false),
    ]));
    let tags = ListArray::new(
        Arc::new(Field::new("element", DataType::Utf8, true)),
        OffsetBuffer::new(ScalarBuffer::from(vec![0, 2, 2, 4, 5])),
        Arc::new(StringArray::from(vec![Some("red"), Some("blue"), None, Some("green"), Some("gold")])),
        Some(NullBuffer::from(vec![true, false, true, true])),
    );
    let batch = RecordBatch::try_new(
        Arc::clone(&source_schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 4])),
            Arc::new(tags),
            Arc::new(Int64Array::from(vec![10, 20, 30, 40])),
        ],
    )
    .unwrap();
    let file = File::create(&path).unwrap();
    let mut writer =
        ArrowWriter::try_new(file, Arc::clone(&source_schema), Some(WriterProperties::builder().build())).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();

    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&source_schema),
        vec!["id".to_string(), "tags".to_string(), "payload".to_string()],
    );
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("tags", DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))), true),
    ]));
    let filter = Expr::binary(Expr::column("payload"), BinaryOp::GtEq, Expr::literal(LiteralValue::Int64(30)));
    let projection = projection_outputs(vec![Expr::column("id"), Expr::column("tags")], Arc::clone(&output_schema));

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let filter = builder.add_filter(source, filter, Arc::clone(&source_schema));
    let projection = builder.add_projection(filter, projection);
    let plan = builder.finish_with_host_sink_schema(projection, Arc::clone(&output_schema));
    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime)).unwrap();

    let result = execute_query_plan(&context, &plan).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize list output");
    };
    assert_eq!(batch.schema(), output_schema);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[3, 4]);
    let tags = batch.column(1).as_any().downcast_ref::<ListArray>().unwrap();
    assert_string_list_value(tags, 0, &[None, Some("green")]);
    assert_string_list_value(tags, 1, &[Some("gold")]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_local_parquet_source_carries_large_list_large_utf8_to_host_output() {
    let dir = unique_temp_dir("nexus-query-engine-native-large-list-utf8");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let source_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("tags", DataType::LargeList(Arc::new(Field::new("element", DataType::LargeUtf8, true))), true),
    ]));
    let tags = LargeListArray::new(
        Arc::new(Field::new("element", DataType::LargeUtf8, true)),
        OffsetBuffer::new(ScalarBuffer::from(vec![0i64, 1, 3])),
        Arc::new(LargeStringArray::from(vec![Some("north"), None, Some("south")])),
        None,
    );
    let batch =
        RecordBatch::try_new(Arc::clone(&source_schema), vec![Arc::new(Int32Array::from(vec![1, 2])), Arc::new(tags)])
            .unwrap();
    let file = File::create(&path).unwrap();
    let mut writer =
        ArrowWriter::try_new(file, Arc::clone(&source_schema), Some(WriterProperties::builder().build())).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();

    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&source_schema),
        vec!["id".to_string(), "tags".to_string()],
    );
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let plan = builder.finish_with_host_sink(source);
    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime)).unwrap();

    let result = execute_query_plan(&context, &plan).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize large-list output");
    };
    assert_eq!(batch.schema(), source_schema);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[1, 2]);
    let tags = batch.column(1).as_any().downcast_ref::<LargeListArray>().unwrap();
    assert_large_string_list_value(tags, 0, &[Some("north")]);
    assert_large_string_list_value(tags, 1, &[None, Some("south")]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_query_executes_union_over_gpu_resident_inputs() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int64, false),
    ]));
    let left_batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(vec![1, 2])), Arc::new(Int64Array::from(vec![10, 20]))],
    )
    .unwrap();
    let right_batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(vec![3])), Arc::new(Int64Array::from(vec![30]))],
    )
    .unwrap();

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("left", Arc::clone(&schema))));
    let right = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("right", Arc::clone(&schema))));
    let union = builder.add_union(vec![left, right], Arc::clone(&schema));
    let plan = builder.finish_with_host_sink(union);
    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime)).unwrap();
    let mut gpu_inputs = GpuResidentInputMap::new();
    gpu_inputs.insert(
        "left".to_string(),
        GpuResultHandle::new(
            &context,
            GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &left_batch).unwrap()),
            Arc::clone(&schema),
        )
        .unwrap(),
    );
    gpu_inputs.insert(
        "right".to_string(),
        GpuResultHandle::new(
            &context,
            GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &right_batch).unwrap()),
            Arc::clone(&schema),
        )
        .unwrap(),
    );

    let result = execute_query_plan_with_gpu_inputs(&context, &plan, &gpu_inputs).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 3);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[1, 2, 3]);
    assert_eq!(int64_array(batch, 1).values().as_ref(), &[10, 20, 30]);
}

#[test]
fn bounded_native_query_executes_union_over_gpu_resident_inputs() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int64, false),
    ]));
    let left_batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(vec![1, 2])), Arc::new(Int64Array::from(vec![10, 20]))],
    )
    .unwrap();
    let right_batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(vec![3])), Arc::new(Int64Array::from(vec![30]))],
    )
    .unwrap();

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("left", Arc::clone(&schema))));
    let right = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("right", Arc::clone(&schema))));
    let union = builder.add_union(vec![left, right], Arc::clone(&schema));
    let plan = builder.finish_with_host_sink(union);
    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 64 * 1024 * 1024,
            max_source_chunk_bytes: 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .build()
        .unwrap();
    let mut gpu_inputs = GpuResidentInputMap::new();
    gpu_inputs.insert(
        "left".to_string(),
        GpuResultHandle::new(
            &context,
            GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &left_batch).unwrap()),
            Arc::clone(&schema),
        )
        .unwrap(),
    );
    gpu_inputs.insert(
        "right".to_string(),
        GpuResultHandle::new(
            &context,
            GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &right_batch).unwrap()),
            Arc::clone(&schema),
        )
        .unwrap(),
    );

    let result = execute_query_plan_with_gpu_inputs(&context, &plan, &gpu_inputs).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 3);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[1, 2, 3]);
    assert_eq!(int64_array(batch, 1).values().as_ref(), &[10, 20, 30]);
}

#[test]
fn bounded_native_query_executes_join_over_gpu_resident_inputs() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let left_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("lval", DataType::Int64, false),
    ]));
    let right_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("rval", DataType::Int64, false),
    ]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("lval", DataType::Int64, false),
        Field::new("rval", DataType::Int64, false),
    ]));
    let left_batch = RecordBatch::try_new(
        Arc::clone(&left_schema),
        vec![Arc::new(Int32Array::from(vec![1, 2, 3])), Arc::new(Int64Array::from(vec![10, 20, 30]))],
    )
    .unwrap();
    let right_batch = RecordBatch::try_new(
        Arc::clone(&right_schema),
        vec![Arc::new(Int32Array::from(vec![2, 3, 4])), Arc::new(Int64Array::from(vec![200, 300, 400]))],
    )
    .unwrap();

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("left", Arc::clone(&left_schema))));
    let right = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("right", Arc::clone(&right_schema))));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                nexus_query_engine::expr::ColumnRef::input(0, "id", "left_id"),
                nexus_query_engine::expr::ColumnRef::input(1, "id", "right_id"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(nexus_query_engine::expr::ColumnRef::input(0, "id", "id"), "id"),
            JoinOutputColumn::left(nexus_query_engine::expr::ColumnRef::input(0, "lval", "lval"), "lval"),
            JoinOutputColumn::right(nexus_query_engine::expr::ColumnRef::input(1, "rval", "rval"), "rval"),
        ])
        .with_null_equality(JoinNullEquality::NullsDoNotMatch)
        .with_build_side(JoinSide::Right),
        Arc::clone(&output_schema),
    );
    let sort = builder.add_sort(
        join,
        SortSpec::new(vec![SortExpr::new(Expr::column("id"), false, false)]),
        Arc::clone(&output_schema),
    );
    let plan = builder.finish_with_host_sink(sort);
    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 64 * 1024 * 1024,
            max_source_chunk_bytes: 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .build()
        .unwrap();
    let mut gpu_inputs = GpuResidentInputMap::new();
    gpu_inputs.insert(
        "left".to_string(),
        GpuResultHandle::new(
            &context,
            GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &left_batch).unwrap()),
            Arc::clone(&left_schema),
        )
        .unwrap(),
    );
    gpu_inputs.insert(
        "right".to_string(),
        GpuResultHandle::new(
            &context,
            GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &right_batch).unwrap()),
            Arc::clone(&right_schema),
        )
        .unwrap(),
    );

    let result = execute_query_plan_with_gpu_inputs(&context, &plan, &gpu_inputs).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize joined rows");
    };
    assert_eq!(batch.num_rows(), 2);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[2, 3]);
    assert_eq!(int64_array(batch, 1).values().as_ref(), &[20, 30]);
    assert_eq!(int64_array(batch, 2).values().as_ref(), &[200, 300]);
}

fn write_nullable_key_payload_parquet(runtime: &cudf::Runtime, path: &Path) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, true),
        Field::new("payload", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![Some(1), Some(2), Some(1), None, Some(3), None])),
            Arc::new(Int64Array::from(vec![5, 10, 20, 30, 40, 50])),
        ],
    )
    .unwrap();
    DataFrame::from_record_batch(runtime, &batch).unwrap().write_parquet(path).unwrap();
    schema
}

fn write_key_payload_row_grouped_parquet(path: &Path, max_row_group_rows: usize) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(vec![1, 2, 1, 3])), Arc::new(Int64Array::from(vec![5, 10, 20, 40]))],
    )
    .unwrap();
    write_record_batch_with_row_group_rows(path, Arc::clone(&schema), batch, max_row_group_rows);
    schema
}

fn write_equality_delete_key_parquet(runtime: &cudf::Runtime, path: &Path, keys: &[Option<i32>]) {
    let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, true)]));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from(keys.to_vec()))]).unwrap();
    DataFrame::from_record_batch(runtime, &batch).unwrap().write_parquet(path).unwrap();
}

fn write_equality_delete_i64_key_parquet(runtime: &cudf::Runtime, path: &Path, keys: &[Option<i64>]) {
    let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int64, true)]));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int64Array::from(keys.to_vec()))]).unwrap();
    DataFrame::from_record_batch(runtime, &batch).unwrap().write_parquet(path).unwrap();
}

fn write_position_delete_parquet(path: &Path, data_file_path: &str, positions: &[i64], other_positions: &[i64]) {
    let row_count = positions.len().saturating_add(other_positions.len()).max(1);
    write_position_delete_parquet_with_row_group_rows(path, data_file_path, positions, other_positions, row_count);
}

fn write_position_delete_parquet_with_row_group_rows(
    path: &Path,
    data_file_path: &str,
    positions: &[i64],
    other_positions: &[i64],
    max_row_group_rows: usize,
) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("file_path", DataType::Utf8, false),
        Field::new("pos", DataType::Int64, false),
    ]));
    let mut file_paths = vec![data_file_path.to_string(); positions.len()];
    file_paths.extend(std::iter::repeat_n("file:/tmp/other.parquet".to_string(), other_positions.len()));
    let mut all_positions = positions.to_vec();
    all_positions.extend_from_slice(other_positions);
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(StringArray::from(file_paths)), Arc::new(Int64Array::from(all_positions))],
    )
    .unwrap();
    let file = File::create(path).unwrap();
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(max_row_group_rows)).build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

fn assert_string_list_value(array: &ListArray, index: usize, expected: &[Option<&str>]) {
    let values = array.value(index);
    let strings = values.as_any().downcast_ref::<StringArray>().unwrap();
    assert_eq!(strings.len(), expected.len());
    for (value_index, expected_value) in expected.iter().enumerate() {
        match expected_value {
            Some(expected_value) => assert_eq!(strings.value(value_index), *expected_value),
            None => assert!(strings.is_null(value_index), "value {value_index} should be null"),
        }
    }
}

fn assert_large_string_list_value(array: &LargeListArray, index: usize, expected: &[Option<&str>]) {
    let values = array.value(index);
    let strings = values.as_any().downcast_ref::<LargeStringArray>().unwrap();
    assert_eq!(strings.len(), expected.len());
    for (value_index, expected_value) in expected.iter().enumerate() {
        match expected_value {
            Some(expected_value) => assert_eq!(strings.value(value_index), *expected_value),
            None => assert!(strings.is_null(value_index), "value {value_index} should be null"),
        }
    }
}

#[test]
fn native_query_executes_literal_batch_source_with_offset_only_limit() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("label", DataType::Utf8, true),
    ]));
    let input_batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(StringArray::from(vec![Some("one"), None, Some("three")])),
        ],
    )
    .unwrap();

    let mut builder = QueryPlanBuilder::new();
    let source =
        builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(Arc::clone(&schema), vec![input_batch])));
    let limit = builder.add_limit(source, LimitSpec::offset(1), Arc::clone(&schema));
    let plan = builder.finish_with_host_sink(limit);

    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime)).unwrap();
    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.input_batches, 1);
    assert_eq!(metrics.input_rows, 3);
    assert_eq!(metrics.boundary.host_to_device_uploads, 1);
    assert_eq!(metrics.boundary.device_to_host_materializations, 1);
    assert_eq!(metrics.boundary.device_to_host_materialized_rows, 2);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 2);
    assert_eq!(batch.num_columns(), 2);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[2, 3]);
    assert!(string_array(batch, 1).is_null(0));
    assert_eq!(string_array(batch, 1).value(1), "three");
}

#[test]
fn native_literal_batch_source_preserves_batch_order() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    let first_batch =
        RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from(vec![10, 11]))]).unwrap();
    let second_batch =
        RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from(vec![20, 21, 22]))]).unwrap();

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(
        Arc::clone(&schema),
        vec![first_batch, second_batch],
    )));
    let plan = builder.finish_with_host_sink(source);

    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime)).unwrap();
    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.input_batches, 2);
    assert_eq!(metrics.input_rows, 5);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 5);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[10, 11, 20, 21, 22]);
}

#[test]
fn native_local_parquet_reader_predicate_coerces_integer_literal_to_float_column_type() {
    let dir = unique_temp_dir("nexus-query-engine-native-reader-float-integer-predicate");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("amount", DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&input_schema),
        vec![Arc::new(Int32Array::from(vec![1, 2, 3])), Arc::new(Float64Array::from(vec![10.5, 20.0, 30.5]))],
    )
    .unwrap();
    DataFrame::from_record_batch(&runtime, &batch).unwrap().write_parquet(&path).unwrap();

    let predicate = Expr::binary(Expr::column("amount"), BinaryOp::Eq, Expr::literal(LiteralValue::Int32(20)));
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["order_id".to_string(), "amount".to_string()],
    )
    .with_reader_predicate(predicate);
    assert!(source.reader_predicate.as_ref().is_some_and(|predicate| source.reader_predicate_is_supported(predicate)));

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let plan = builder.finish_with_host_sink(source);

    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime)).unwrap();
    let result = execute_query_plan(&context, &plan).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 1);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[2]);
    assert_eq!(float64_array(batch, 1).values().as_ref(), &[20.0]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_local_parquet_reader_predicate_uses_sql_null_logical_or() {
    let dir = unique_temp_dir("nexus-query-engine-native-reader-null-logical-or");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("dmail", DataType::Utf8, true),
        Field::new("email", DataType::Utf8, true),
        Field::new("tv", DataType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5, 6])),
            Arc::new(StringArray::from(vec![Some("Y"), Some("N"), Some("N"), None, Some("N"), None])),
            Arc::new(StringArray::from(vec![None, Some("Y"), None, Some("Y"), Some("N"), None])),
            Arc::new(StringArray::from(vec![Some("N"), None, Some("N"), Some("N"), Some("N"), None])),
        ],
    )
    .unwrap();
    DataFrame::from_record_batch(&runtime, &batch).unwrap().write_parquet(&path).unwrap();

    let y = || Expr::literal(LiteralValue::Utf8("Y".to_string()));
    let dmail_is_y = Expr::binary(Expr::column("dmail"), BinaryOp::Eq, y());
    let email_is_y = Expr::binary(Expr::column("email"), BinaryOp::Eq, y());
    let tv_is_y = Expr::binary(Expr::column("tv"), BinaryOp::Eq, y());
    let predicate = Expr::binary(Expr::binary(dmail_is_y, BinaryOp::Or, email_is_y), BinaryOp::Or, tv_is_y);
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&schema),
        vec!["key".to_string(), "dmail".to_string(), "email".to_string(), "tv".to_string()],
    )
    .with_column_roles(vec!["key".to_string()], vec!["dmail".to_string(), "email".to_string(), "tv".to_string()])
    .with_reader_predicate(predicate);

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let plan = builder.finish_with_host_sink(source);

    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime)).unwrap();
    let result = execute_query_plan(&context, &plan).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 3);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[1, 2, 4]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_filter_predicate_uses_where_semantics_for_nullable_and_or_in_lists() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("a", DataType::Boolean, true),
        Field::new("b", DataType::Boolean, true),
        Field::new("label", DataType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5, 6])),
            Arc::new(BooleanArray::from(vec![Some(true), Some(true), Some(false), None, None, None])),
            Arc::new(BooleanArray::from(vec![Some(true), None, None, Some(true), Some(false), None])),
            Arc::new(StringArray::from(vec![Some("A"), Some("C"), None, Some("B"), Some("D"), Some("A")])),
        ],
    )
    .unwrap();

    assert_eq!(
        execute_gpu_resident_filter_ids(
            &runtime,
            Arc::clone(&schema),
            &batch,
            Expr::binary(Expr::column("a"), BinaryOp::And, Expr::column("b"))
        ),
        vec![1]
    );
    assert_eq!(
        execute_gpu_resident_filter_ids(
            &runtime,
            Arc::clone(&schema),
            &batch,
            Expr::binary(Expr::column("a"), BinaryOp::Or, Expr::column("b"))
        ),
        vec![1, 2, 4]
    );
    assert_eq!(
        execute_gpu_resident_filter_ids(
            &runtime,
            Arc::clone(&schema),
            &batch,
            Expr::in_list(
                Expr::column("label"),
                vec![
                    Expr::literal(LiteralValue::Utf8("A".to_string())),
                    Expr::literal(LiteralValue::Utf8("B".to_string())),
                ],
            ),
        ),
        vec![1, 4, 6]
    );
    assert_eq!(
        execute_gpu_resident_filter_ids(
            &runtime,
            Arc::clone(&schema),
            &batch,
            Expr::not_in_list(
                Expr::column("label"),
                vec![
                    Expr::literal(LiteralValue::Utf8("A".to_string())),
                    Expr::literal(LiteralValue::Utf8("B".to_string())),
                ],
            ),
        ),
        vec![2, 5]
    );
}

#[test]
fn native_filter_predicate_executes_large_numeric_literal_in_list() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let schema =
        Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false), Field::new("key", DataType::Int64, true)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5, 6])),
            Arc::new(Int64Array::from(vec![Some(2), Some(3), Some(128), None, Some(250), Some(260)])),
        ],
    )
    .unwrap();
    let list = (0..129).map(|value| Expr::literal(LiteralValue::Int64(value * 2))).collect();

    assert_eq!(
        execute_gpu_resident_filter_ids(
            &runtime,
            Arc::clone(&schema),
            &batch,
            Expr::in_list(Expr::column("key"), list)
        ),
        vec![1, 3, 5]
    );
}

fn execute_gpu_resident_filter_ids(
    runtime: &Arc<cudf::Runtime>,
    schema: Arc<Schema>,
    batch: &RecordBatch,
    predicate: Expr,
) -> Vec<i32> {
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", Arc::clone(&schema))));
    let filter = builder.add_filter(source, predicate, Arc::clone(&schema));
    let plan = builder.finish_with_host_sink(filter);
    let context = ExecutionContext::from_shared_runtime(Arc::clone(runtime)).unwrap();
    let mut gpu_inputs = GpuResidentInputMap::new();
    gpu_inputs.insert(
        "input".to_string(),
        GpuResultHandle::new(
            &context,
            GpuDataFrame::new(DataFrame::from_record_batch(runtime, batch).unwrap()),
            Arc::clone(&schema),
        )
        .unwrap(),
    );

    let result = execute_query_plan_with_gpu_inputs(&context, &plan, &gpu_inputs).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    int32_array(batch, 0).values().to_vec()
}

#[test]
fn native_local_parquet_source_scan_limit_applies_before_parent_operators() {
    let dir = unique_temp_dir("nexus-query-engine-native-source-scan-limit");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_input_parquet(&runtime, &path);
    let projected_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int64, false),
    ]));

    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["key".to_string(), "payload".to_string()],
    )
    .with_scan_limit(Some(2));
    let projection = vec![Expr::column("key"), Expr::column("payload")];

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let projection = builder.add_projection(source, projection_outputs(projection, projected_schema));
    let plan = builder.finish_with_host_sink(projection);

    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime)).unwrap();
    let result = execute_query_plan(&context, &plan).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 2);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[1, 2]);
    assert_eq!(int64_array(batch, 1).values().as_ref(), &[5, 10]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_local_parquet_source_materializes_missing_nullable_columns() {
    let dir = unique_temp_dir("nexus-query-engine-native-missing-nullable-source-columns");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    write_input_parquet(&runtime, &path);

    let source_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("missing_payload", DataType::Int64, true),
    ]));
    let source = LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new(&path)
                .with_file_size_bytes(fs::metadata(&path).unwrap().len())
                .with_row_group_rows(vec![4]),
        ],
        Arc::clone(&source_schema),
        vec!["key".to_string()],
    )
    .with_missing_columns(vec![MissingColumn::new("missing_payload", DataType::Int64, true)])
    .with_column_roles(vec!["key".to_string(), "missing_payload".to_string()], Vec::new());

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let plan = builder.finish_with_host_sink(source);

    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime)).unwrap();
    let result = execute_query_plan(&context, &plan).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 4);
    assert_eq!(batch.num_columns(), 2);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[1, 2, 1, 3]);
    let missing = batch.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
    assert_eq!(missing.null_count(), 4);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_local_parquet_source_materializes_missing_nullable_columns() {
    let dir = unique_temp_dir("nexus-query-engine-bounded-missing-nullable-source-columns");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    write_input_parquet(&runtime, &path);

    let source_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("missing_payload", DataType::Int64, true),
    ]));
    let source = LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new(&path)
                .with_file_size_bytes(fs::metadata(&path).unwrap().len())
                .with_row_group_bytes(vec![4096])
                .with_row_group_rows(vec![4]),
        ],
        Arc::clone(&source_schema),
        vec!["key".to_string()],
    )
    .with_missing_columns(vec![MissingColumn::new("missing_payload", DataType::Int64, true)])
    .with_column_roles(vec!["key".to_string(), "missing_payload".to_string()], Vec::new());

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let plan = builder.finish_with_host_sink(source);

    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 64 * 1024 * 1024,
            max_source_chunk_bytes: 16 * 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 2,
            min_retry_chunk_row_groups: 1,
        })
        .build()
        .unwrap();
    let result = execute_query_plan(&context, &plan).unwrap();
    assert_eq!(result.metrics().source_chunks_committed, 1);
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 4);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[1, 2, 1, 3]);
    let missing = batch.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
    assert_eq!(missing.null_count(), 4);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_generic_local_parquet_pipeline_uses_required_only_source_scheduler() {
    let dir = unique_temp_dir("nexus-query-engine-bounded-generic-source-scheduler");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_row_grouped_input_parquet(&path);

    let projected_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("x00", DataType::Int64, false),
    ]));
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["key".to_string(), "m00".to_string(), "m01".to_string(), "payload".to_string()],
    )
    .with_column_roles(vec!["key".to_string(), "m00".to_string(), "m01".to_string()], vec!["payload".to_string()]);
    let filter = Expr::binary(Expr::column("payload"), BinaryOp::GtEq, Expr::literal(LiteralValue::Int64(10)));
    let projection =
        vec![Expr::column("key"), Expr::binary(Expr::column("m00"), BinaryOp::Add, Expr::column("m01")).alias("x00")];
    let sort = SortSpec::new(vec![
        SortExpr::new(Expr::column("key"), false, false),
        SortExpr::new(Expr::column("x00"), true, false),
    ]);

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let filter = builder.add_filter(source, filter, Arc::clone(&input_schema));
    let projection = builder.add_projection(filter, projection_outputs(projection, Arc::clone(&projected_schema)));
    let sort = builder.add_sort(projection, sort, Arc::clone(&projected_schema));
    let limit = builder.add_limit(sort, LimitSpec::fetch(4), projected_schema);
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
        .with_source_scheduler_policy(SourceSchedulerPolicy::new(3))
        .build()
        .unwrap();

    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.source_chunks_planned, 3);
    assert_eq!(metrics.source_chunks_attempted, 3);
    assert_eq!(metrics.source_chunks_committed, 3);
    assert_eq!(metrics.source_read_prefetches_started, 0);
    assert_eq!(metrics.source_read_prefetches_completed, 0);
    assert_eq!(metrics.source_read_prefetches_skipped, 0);
    assert_eq!(metrics.source_read_prefetches_abandoned, 0);
    let production_prefetches = metrics
        .reservation_events
        .iter()
        .filter(|event| {
            event.outcome.as_str() == "granted"
                && matches!(event.memory_boundary.as_str(), "source_chunk_prefetch" | "join_source_read_prefetch")
        })
        .count();
    assert_eq!(production_prefetches, 0);
    assert!(metrics.reservation_events.iter().any(|event| {
        event.outcome.as_str() == "granted"
            && event.memory_boundary == "bounded_generic_source_read"
            && event.action == MemoryReservationEventAction::Reserve
    }));
    assert!(
        metrics.reservation_events.iter().all(|event| {
            !(event.memory_boundary == "bounded_generic_source_read"
                && event.action == MemoryReservationEventAction::Grow)
        }),
        "source-adjacent generic filtering/projection should not grow raw source-read reservations to observed bytes"
    );
    assert!(
        metrics.reservation_events.iter().any(|event| {
            event.memory_boundary == "bounded_generic_source_filter_output"
                && event.action == MemoryReservationEventAction::Transfer
        }),
        "source-adjacent generic filtering should transfer ownership to the filtered output boundary"
    );
    let first_filter_transfer = metrics
        .reservation_events
        .iter()
        .position(|event| {
            event.memory_boundary == "bounded_generic_source_filter_output"
                && event.action == MemoryReservationEventAction::Transfer
        })
        .expect("source-adjacent generic filtering should transfer each chunk before source concat");
    let source_read_reserves_before_first_filter = metrics.reservation_events[..first_filter_transfer]
        .iter()
        .filter(|event| {
            event.outcome.as_str() == "granted"
                && event.memory_boundary == "bounded_generic_source_read"
                && event.action == MemoryReservationEventAction::Reserve
        })
        .count();
    assert_eq!(
        source_read_reserves_before_first_filter, 1,
        "generic source-adjacent streaming should filter/project the first chunk before reading later chunks"
    );
    assert!(
        metrics.reservation_events.iter().any(|event| {
            event.memory_boundary == "bounded_generic_source_projection_output"
                && event.action == MemoryReservationEventAction::Transfer
        }),
        "source-adjacent generic projection should transfer ownership to the projected output boundary"
    );
    assert!(
        metrics.reservation_events.iter().any(|event| {
            event.memory_boundary == "bounded_generic_source_concat_output"
                && event.action == MemoryReservationEventAction::Reserve
        }),
        "multi-chunk generic source concat output should be traced separately from raw source reads"
    );

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 4);
    assert_eq!(batch.num_columns(), 2);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[1, 1, 2, 2]);
    assert_eq!(int64_array(batch, 1).values().as_ref(), &[303, 101, 505, 202]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_generic_local_parquet_source_batches_adjacent_files() {
    let dir = unique_temp_dir("nexus-query-engine-bounded-generic-source-batch");
    let paths = [dir.join("input-0.parquet"), dir.join("input-1.parquet"), dir.join("input-2.parquet")];
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    write_input_parquet(&runtime, &paths[0]);
    write_input_parquet(&runtime, &paths[1]);
    write_input_parquet(&runtime, &paths[2]);
    let source_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int64, false),
    ]));

    let files = paths
        .iter()
        .map(|path| {
            LocalParquetFile::new(path)
                .with_file_size_bytes(fs::metadata(path).unwrap().len())
                .with_row_group_rows(vec![4])
        })
        .collect::<Vec<_>>();
    let source = LocalParquetSource::from_files(files, source_schema, vec!["key".to_string(), "payload".to_string()]);

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let plan = builder.finish_with_host_sink(source);
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
    assert_eq!(metrics.source_chunks_attempted, 3);
    assert_eq!(metrics.source_chunks_committed, 3);
    assert_eq!(metrics.row_groups_planned, 3);
    assert_eq!(metrics.row_groups_committed, 3);
    assert_eq!(metrics.source_read_runtime_rows.len(), 1);
    let source_read_row = &metrics.source_read_runtime_rows[0];
    assert_eq!(source_read_row.read_kind, "chunk_batch");
    assert_eq!(source_read_row.read_strategy, "local_kvikio");
    assert_eq!(source_read_row.chunk_count, 3);
    assert_eq!(source_read_row.file_count, 3);
    assert_eq!(source_read_row.row_group_count, 3);
    assert_eq!(source_read_row.read_columns_per_file, 2);
    assert_eq!(source_read_row.read_columns_total, 6);
    assert_eq!(source_read_row.parquet_read_evidence.input_kind, "local_path_multi");
    assert_eq!(source_read_row.parquet_read_evidence.remote_read_mode, "none");
    assert_eq!(source_read_row.parquet_read_evidence.backend, "libcudf_path_reader");
    assert_eq!(source_read_row.parquet_read_evidence.fallback_reason, "not_applicable");

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 12);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[1, 2, 1, 3, 1, 2, 1, 3, 1, 2, 1, 3]);
    assert_eq!(int64_array(batch, 1).values().as_ref(), &[5, 10, 20, 20, 5, 10, 20, 20, 5, 10, 20, 20]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_generic_local_parquet_source_scan_limit_trims_to_required_row_group_prefix() {
    let dir = unique_temp_dir("nexus-query-engine-bounded-source-scan-limit");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let _input_schema = write_row_grouped_input_parquet(&path);
    let source_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int64, false),
    ]));

    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        source_schema,
        vec!["key".to_string(), "payload".to_string()],
    )
    .with_scan_limit(Some(3));

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let plan = builder.finish_with_host_sink(source);

    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 8 * 1024 * 1024 * 1024,
            max_source_chunk_bytes: 8 * 1024 * 1024 * 1024,
            max_row_groups_per_chunk: 8,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .with_source_scheduler_policy(SourceSchedulerPolicy::new(3))
        .build()
        .unwrap();

    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.source_chunks_planned, 1);
    assert_eq!(metrics.source_chunks_attempted, 1);
    assert_eq!(metrics.source_chunks_committed, 1);
    assert_eq!(metrics.row_groups_planned, 2);
    assert_eq!(metrics.row_groups_committed, 2);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 3);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[1, 2, 1]);
    assert_eq!(int64_array(batch, 1).values().as_ref(), &[10, 10, 20]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_local_parquet_source_cache_reuses_equivalent_relowered_scan_contract() {
    let dir = unique_temp_dir("nexus-query-engine-bounded-source-cache-reuse");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let _input_schema = write_row_grouped_input_parquet(&path);
    let source_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int64, false),
    ]));
    let cache = Arc::new(GpuLocalParquetSourceCache::new(64 * 1024 * 1024));
    let file_size = fs::metadata(&path).unwrap().len();

    let build_plan = || {
        let source = LocalParquetSource::from_files(
            vec![LocalParquetFile::new(&path).with_stable_id("stable-input").with_file_size_bytes(file_size)],
            Arc::clone(&source_schema),
            vec!["key".to_string(), "payload".to_string()],
        );

        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(source));
        builder.finish_with_host_sink(source)
    };
    let build_context = |runtime: &Arc<cudf::Runtime>| {
        ExecutionContext::builder()
            .with_shared_runtime(Arc::clone(runtime))
            .with_memory_control_policy(MemoryControlPolicy::Bounded {
                device_budget_bytes: 8 * 1024 * 1024 * 1024,
                max_source_chunk_bytes: 1024 * 1024,
                max_row_groups_per_chunk: 1,
                max_retry_attempts: 1,
                min_retry_chunk_row_groups: 1,
            })
            .with_local_parquet_source_cache(Arc::clone(&cache))
            .build()
            .unwrap()
    };

    let first = execute_query_plan(&build_context(&runtime), &build_plan()).unwrap();
    let first_metrics = first.metrics();
    assert_eq!(first_metrics.source_chunks_planned, 3);
    assert_eq!(first_metrics.source_cache_hits, 0);
    assert_eq!(first_metrics.source_cache_misses, 3);
    assert_eq!(first_metrics.source_cache_inserts, 3);
    assert_eq!(first_metrics.source_cache_insert_skipped, 0);
    assert!(first_metrics.source_cache_retained_bytes > 0);
    assert_eq!(cache.len().unwrap(), 3);
    let retained_bytes = cache.retained_device_bytes().unwrap();
    assert!(retained_bytes > 0);

    let second = execute_query_plan(&build_context(&runtime), &build_plan()).unwrap();
    let second_metrics = second.metrics();
    assert_eq!(second_metrics.source_chunks_planned, 3);
    assert_eq!(second_metrics.source_cache_hits, 3);
    assert_eq!(second_metrics.source_cache_misses, 0);
    assert_eq!(second_metrics.source_cache_inserts, 0);
    assert_eq!(second_metrics.source_cache_insert_skipped, 0);
    assert_eq!(second_metrics.source_cache_retained_bytes, retained_bytes);
    assert!(second_metrics.reservation_events.iter().any(|event| {
        event.outcome.as_str() == "granted"
            && event.reservation_kind == ReservationKind::SourceCacheRetained
            && event.memory_boundary == "source_cache_retained"
            && event.requested_bytes == retained_bytes
    }));

    let QueryOutput::HostArrow(batch) = second.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 6);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[1, 2, 1, 3, 2, 3]);
    assert_eq!(int64_array(batch, 1).values().as_ref(), &[10, 10, 20, 20, 30, 5]);

    let other_runtime = Arc::new(cudf::Runtime::new().unwrap());
    let third = execute_query_plan(&build_context(&other_runtime), &build_plan()).unwrap();
    let third_metrics = third.metrics();
    assert_eq!(third_metrics.source_cache_hits, 0);
    assert_eq!(third_metrics.source_cache_misses, 3);
    assert_eq!(third_metrics.source_cache_inserts, 3);
    assert_eq!(cache.len().unwrap(), 6);
    let retained_after_other_runtime = cache.retained_device_bytes().unwrap();
    assert_eq!(retained_after_other_runtime, retained_bytes.saturating_mul(2));

    let fourth = execute_query_plan(&build_context(&runtime), &build_plan()).unwrap();
    let fourth_metrics = fourth.metrics();
    assert_eq!(fourth_metrics.source_cache_hits, 3);
    assert_eq!(fourth_metrics.source_cache_misses, 0);
    assert_eq!(fourth_metrics.source_cache_inserts, 0);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn whole_source_local_parquet_source_cache_uses_active_grant_capacity() {
    let dir = unique_temp_dir("nexus-query-engine-whole-source-cache-grant");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    write_row_grouped_input_parquet(&path);
    let source_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int64, false),
    ]));
    let governor = MemoryGovernor::new(runtime.device().ordinal(), 64 * 1024 * 1024).unwrap();
    let grant = governor.request(GrantRequest::new(200, governor.capacity_bytes(), governor.capacity_bytes())).unwrap();
    assert_eq!(governor.free_bytes().unwrap(), 0);
    let cache = Arc::new(GpuLocalParquetSourceCache::new_governor_backed(64 * 1024 * 1024, governor.clone()));
    GpuLocalParquetSourceCache::register_governor_pressure_observer(&cache).unwrap();
    let file_size = fs::metadata(&path).unwrap().len();

    let build_plan = || {
        let source = LocalParquetSource::from_files(
            vec![LocalParquetFile::new(&path).with_stable_id("stable-input").with_file_size_bytes(file_size)],
            Arc::clone(&source_schema),
            vec!["key".to_string(), "payload".to_string()],
        );

        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(source));
        builder.finish_with_host_sink(source)
    };
    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_grant(grant)
        .with_local_parquet_source_cache(Arc::clone(&cache))
        .build()
        .unwrap();

    let cold = execute_query_plan(&context, &build_plan()).unwrap();
    assert_eq!(cold.metrics().source_cache_hits, 0);
    assert_eq!(cold.metrics().source_cache_misses, 1);
    assert_eq!(cold.metrics().source_cache_inserts, 1);
    let retained_bytes = cache.retained_device_bytes().unwrap();
    assert!(retained_bytes > 0);
    let report = governor.report().unwrap();
    assert_eq!(report.evictable_bytes, retained_bytes);
    assert_eq!(report.outstanding_bytes, governor.capacity_bytes());
    assert_eq!(report.free_bytes, 0);

    let warm = execute_query_plan(&context, &build_plan()).unwrap();
    assert_eq!(warm.metrics().source_cache_hits, 1);
    assert_eq!(warm.metrics().source_cache_misses, 0);
    assert_eq!(warm.metrics().source_cache_inserts, 0);
    assert_eq!(cache.retained_device_bytes().unwrap(), retained_bytes);

    let QueryOutput::HostArrow(batch) = warm.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 6);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[1, 2, 1, 3, 2, 3]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn whole_source_local_parquet_source_cache_miss_projection_pins_gpu_result_until_drop() {
    let dir = unique_temp_dir("nexus-query-engine-whole-source-cache-projection-pin");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    write_row_grouped_input_parquet(&path);
    let source_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int64, false),
    ]));
    let output_schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, false)]));
    let governor = MemoryGovernor::new(runtime.device().ordinal(), 64 * 1024 * 1024).unwrap();
    let cache = Arc::new(GpuLocalParquetSourceCache::new_governor_backed(64 * 1024 * 1024, governor.clone()));
    GpuLocalParquetSourceCache::register_governor_pressure_observer(&cache).unwrap();
    let file_size = fs::metadata(&path).unwrap().len();

    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_stable_id("stable-input").with_file_size_bytes(file_size)],
        Arc::clone(&source_schema),
        vec!["key".to_string(), "payload".to_string()],
    );
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let projection =
        builder.add_projection(source, projection_outputs(vec![Expr::column("key")], Arc::clone(&output_schema)));
    let plan = builder.finish_gpu_resident(projection);
    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_local_parquet_source_cache(Arc::clone(&cache))
        .build()
        .unwrap();

    let cold = execute_query_plan(&context, &plan).unwrap();
    assert_eq!(cold.metrics().source_cache_hits, 0);
    assert_eq!(cold.metrics().source_cache_misses, 1);
    assert_eq!(cold.metrics().source_cache_inserts, 1);
    let QueryOutput::GpuResident(handle) = cold.output() else {
        panic!("gpu sink should retain the projected cached frame");
    };
    assert_eq!(handle.num_rows(), 6);
    assert_eq!(handle.num_columns(), 1);
    let retained_bytes = cache.retained_device_bytes().unwrap();
    assert!(retained_bytes > 0);
    assert!(
        governor
            .try_request(GrantRequest::new(201, governor.capacity_bytes(), governor.capacity_bytes()))
            .unwrap()
            .is_none(),
        "pressure admission must not evict a cold-miss source-cache entry pinned by projected GPU output"
    );
    assert_eq!(cache.retained_device_bytes().unwrap(), retained_bytes);
    assert_eq!(governor.evictable_bytes().unwrap(), retained_bytes);

    drop(cold);
    let grant = governor
        .try_request(GrantRequest::new(202, governor.capacity_bytes(), governor.capacity_bytes()))
        .unwrap()
        .expect("unpinned cold-miss cache entry should be evicted under pressure");
    assert_eq!(grant.budget_bytes(), governor.capacity_bytes());
    assert_eq!(cache.retained_device_bytes().unwrap(), 0);
    assert_eq!(governor.evictable_bytes().unwrap(), 0);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn whole_source_local_parquet_source_cache_capacity_skip_returns_uncached_result() {
    let dir = unique_temp_dir("nexus-query-engine-whole-source-cache-capacity-skip");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    write_row_grouped_input_parquet(&path);
    let source_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int64, false),
    ]));
    let cache = Arc::new(GpuLocalParquetSourceCache::new(1));
    let file_size = fs::metadata(&path).unwrap().len();
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_stable_id("stable-input").with_file_size_bytes(file_size)],
        Arc::clone(&source_schema),
        vec!["key".to_string(), "payload".to_string()],
    );
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let plan = builder.finish_with_host_sink(source);
    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_local_parquet_source_cache(Arc::clone(&cache))
        .build()
        .unwrap();

    let result = execute_query_plan(&context, &plan).unwrap();
    assert_eq!(result.metrics().source_cache_hits, 0);
    assert_eq!(result.metrics().source_cache_misses, 1);
    assert_eq!(result.metrics().source_cache_inserts, 0);
    assert_eq!(result.metrics().source_cache_insert_skipped, 1);
    assert_eq!(cache.retained_device_bytes().unwrap(), 0);
    assert!(cache.is_empty().unwrap());

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should still materialize the uncached cold-miss result");
    };
    assert_eq!(batch.num_rows(), 6);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[1, 2, 1, 3, 2, 3]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn whole_source_local_parquet_source_cache_single_input_union_pins_gpu_result_until_drop() {
    let dir = unique_temp_dir("nexus-query-engine-whole-source-cache-union-pin");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    write_row_grouped_input_parquet(&path);
    let source_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int64, false),
    ]));
    let governor = MemoryGovernor::new(runtime.device().ordinal(), 64 * 1024 * 1024).unwrap();
    let cache = Arc::new(GpuLocalParquetSourceCache::new_governor_backed(64 * 1024 * 1024, governor.clone()));
    GpuLocalParquetSourceCache::register_governor_pressure_observer(&cache).unwrap();
    let file_size = fs::metadata(&path).unwrap().len();
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_stable_id("stable-input").with_file_size_bytes(file_size)],
        Arc::clone(&source_schema),
        vec!["key".to_string(), "payload".to_string()],
    );
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let union = builder.add_union(vec![source], Arc::clone(&source_schema));
    let plan = builder.finish_gpu_resident(union);
    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_local_parquet_source_cache(Arc::clone(&cache))
        .build()
        .unwrap();

    let cold = execute_query_plan(&context, &plan).unwrap();
    assert_eq!(cold.metrics().source_cache_hits, 0);
    assert_eq!(cold.metrics().source_cache_misses, 1);
    assert_eq!(cold.metrics().source_cache_inserts, 1);
    let QueryOutput::GpuResident(handle) = cold.output() else {
        panic!("gpu sink should retain the union output");
    };
    assert_eq!(handle.num_rows(), 6);
    assert_eq!(handle.num_columns(), 2);
    let retained_bytes = cache.retained_device_bytes().unwrap();
    assert!(retained_bytes > 0);
    assert!(
        governor
            .try_request(GrantRequest::new(203, governor.capacity_bytes(), governor.capacity_bytes()))
            .unwrap()
            .is_none(),
        "pressure admission must not evict a source-cache entry pinned through single-input union"
    );

    drop(cold);
    let grant = governor
        .try_request(GrantRequest::new(204, governor.capacity_bytes(), governor.capacity_bytes()))
        .unwrap()
        .expect("unpinned union output should release source-cache pressure protection");
    assert_eq!(grant.budget_bytes(), governor.capacity_bytes());
    assert_eq!(cache.retained_device_bytes().unwrap(), 0);
    assert_eq!(governor.evictable_bytes().unwrap(), 0);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn whole_source_local_parquet_source_cache_edge_normalize_pins_alias_until_gpu_result_drop() {
    let dir = unique_temp_dir("nexus-query-engine-whole-source-cache-edge-normalize-pin");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    write_row_grouped_input_parquet(&path);
    let source_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int64, false),
    ]));
    let governor = MemoryGovernor::new(runtime.device().ordinal(), 64 * 1024 * 1024).unwrap();
    let cache = Arc::new(GpuLocalParquetSourceCache::new_governor_backed(64 * 1024 * 1024, governor.clone()));
    GpuLocalParquetSourceCache::register_governor_pressure_observer(&cache).unwrap();
    let file_size = fs::metadata(&path).unwrap().len();
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_stable_id("stable-input").with_file_size_bytes(file_size)],
        Arc::clone(&source_schema),
        vec!["key".to_string(), "payload".to_string()],
    );
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let edge_normalize = builder.add_edge_normalize(
        source,
        nexus_query_engine::plan::EdgeNormalizeSpec {
            edge_columns: nexus_query_engine::plan::GraphEdgeColumnBindings::new("key", "payload", None::<&str>),
            add_reverse: false,
            dedup: nexus_query_engine::plan::EdgeDedupStrategy::None,
            weight_combiner: nexus_query_engine::plan::EdgeWeightCombiner::None,
        },
        Arc::clone(&source_schema),
    );
    let plan = builder.finish_gpu_resident(edge_normalize);
    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_local_parquet_source_cache(Arc::clone(&cache))
        .build()
        .unwrap();

    let cold = execute_query_plan(&context, &plan).unwrap();
    assert_eq!(cold.metrics().source_cache_hits, 0);
    assert_eq!(cold.metrics().source_cache_misses, 1);
    assert_eq!(cold.metrics().source_cache_inserts, 1);
    let QueryOutput::GpuResident(handle) = cold.output() else {
        panic!("gpu sink should retain the edge-normalized output");
    };
    assert_eq!(handle.num_rows(), 6);
    assert_eq!(handle.num_columns(), 2);
    let retained_bytes = cache.retained_device_bytes().unwrap();
    assert!(retained_bytes > 0);
    assert!(
        governor
            .try_request(GrantRequest::new(206, governor.capacity_bytes(), governor.capacity_bytes()))
            .unwrap()
            .is_none(),
        "pressure admission must not evict source-cache entry while edge-normalized GPU output aliases it"
    );

    drop(cold);
    let grant = governor
        .try_request(GrantRequest::new(207, governor.capacity_bytes(), governor.capacity_bytes()))
        .unwrap()
        .expect("unpinned edge-normalize source-cache entry should be evicted under pressure");
    assert_eq!(grant.budget_bytes(), governor.capacity_bytes());
    assert_eq!(cache.retained_device_bytes().unwrap(), 0);
    assert_eq!(governor.evictable_bytes().unwrap(), 0);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_local_parquet_source_cache_governor_owns_evictable_bytes() {
    let dir = unique_temp_dir("nexus-query-engine-bounded-source-cache-governor");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let _input_schema = write_row_grouped_input_parquet(&path);
    let source_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int64, false),
    ]));
    let governor = MemoryGovernor::new(runtime.device().ordinal(), 64 * 1024 * 1024).unwrap();
    let cache = Arc::new(GpuLocalParquetSourceCache::new_governor_backed(64 * 1024 * 1024, governor.clone()));
    GpuLocalParquetSourceCache::register_governor_pressure_observer(&cache).unwrap();
    let file_size = fs::metadata(&path).unwrap().len();

    let build_plan = || {
        let source = LocalParquetSource::from_files(
            vec![LocalParquetFile::new(&path).with_stable_id("stable-input").with_file_size_bytes(file_size)],
            Arc::clone(&source_schema),
            vec!["key".to_string(), "payload".to_string()],
        );

        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(source));
        builder.finish_with_host_sink(source)
    };
    let build_context = || {
        ExecutionContext::builder()
            .with_shared_runtime(Arc::clone(&runtime))
            .with_memory_control_policy(MemoryControlPolicy::Bounded {
                device_budget_bytes: 8 * 1024 * 1024 * 1024,
                max_source_chunk_bytes: 1024 * 1024,
                max_row_groups_per_chunk: 1,
                max_retry_attempts: 1,
                min_retry_chunk_row_groups: 1,
            })
            .with_local_parquet_source_cache(Arc::clone(&cache))
            .build()
            .unwrap()
    };

    let first = execute_query_plan(&build_context(), &build_plan()).unwrap();
    assert_eq!(first.metrics().source_cache_hits, 0);
    assert_eq!(first.metrics().source_cache_inserts, 3);
    assert!(
        first.metrics().reservation_events.iter().all(|event| {
            event.reservation_kind != ReservationKind::SourceCacheRetained || event.charged_delta_bytes == 0
        }),
        "governor-backed cache inserts must not charge a per-query source-cache baseline"
    );
    let retained_bytes = cache.retained_device_bytes().unwrap();
    assert!(retained_bytes > 0);
    let report = governor.report().unwrap();
    assert_eq!(report.evictable_bytes, retained_bytes);
    assert_eq!(report.outstanding_bytes, 0);
    assert_eq!(report.free_bytes, report.capacity_bytes.saturating_sub(retained_bytes));

    let second = execute_query_plan(&build_context(), &build_plan()).unwrap();
    assert_eq!(second.metrics().source_cache_hits, 3);
    assert_eq!(second.metrics().source_cache_misses, 0);
    assert_eq!(governor.evictable_bytes().unwrap(), retained_bytes);

    drop(first);
    drop(second);
    let pressure_grant = governor
        .try_request(GrantRequest::new(99, governor.capacity_bytes(), governor.capacity_bytes()))
        .unwrap()
        .expect("admission should evict unpinned source-cache entries");
    assert_eq!(pressure_grant.budget_bytes(), governor.capacity_bytes());
    assert_eq!(cache.retained_device_bytes().unwrap(), 0);
    assert_eq!(governor.evictable_bytes().unwrap(), 0);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_local_parquet_source_cache_does_not_use_active_grant_capacity_for_inserts() {
    let dir = unique_temp_dir("nexus-query-engine-bounded-source-cache-no-grant-fallback");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let _input_schema = write_row_grouped_input_parquet(&path);
    let source_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int64, false),
    ]));
    let governor = MemoryGovernor::new(runtime.device().ordinal(), 64 * 1024 * 1024).unwrap();
    let grant = governor.request(GrantRequest::new(205, governor.capacity_bytes(), governor.capacity_bytes())).unwrap();
    assert_eq!(governor.free_bytes().unwrap(), 0);
    let cache = Arc::new(GpuLocalParquetSourceCache::new_governor_backed(64 * 1024 * 1024, governor.clone()));
    GpuLocalParquetSourceCache::register_governor_pressure_observer(&cache).unwrap();
    let file_size = fs::metadata(&path).unwrap().len();

    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_stable_id("stable-input").with_file_size_bytes(file_size)],
        Arc::clone(&source_schema),
        vec!["key".to_string(), "payload".to_string()],
    );
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let plan = builder.finish_with_host_sink(source);
    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_grant(grant)
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 8 * 1024 * 1024 * 1024,
            max_source_chunk_bytes: 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .with_local_parquet_source_cache(Arc::clone(&cache))
        .build()
        .unwrap();

    let result = execute_query_plan(&context, &plan).unwrap();
    assert_eq!(result.metrics().source_cache_hits, 0);
    assert_eq!(result.metrics().source_cache_misses, 3);
    assert_eq!(result.metrics().source_cache_inserts, 0);
    assert_eq!(result.metrics().source_cache_insert_skipped, 3);
    assert_eq!(cache.retained_device_bytes().unwrap(), 0);
    assert_eq!(governor.evictable_bytes().unwrap(), 0);
    let report = governor.report().unwrap();
    assert_eq!(report.outstanding_bytes, governor.capacity_bytes());
    assert_eq!(report.free_bytes, 0);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize even when bounded source-cache inserts are skipped");
    };
    assert_eq!(batch.num_rows(), 6);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[1, 2, 1, 3, 2, 3]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_local_parquet_source_cache_pressure_skips_gpu_resident_hit_until_drop() {
    let dir = unique_temp_dir("nexus-query-engine-bounded-source-cache-governor-pinned");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let _input_schema = write_row_grouped_input_parquet(&path);
    let source_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int64, false),
    ]));
    let governor = MemoryGovernor::new(runtime.device().ordinal(), 64 * 1024 * 1024).unwrap();
    let cache = Arc::new(GpuLocalParquetSourceCache::new_governor_backed(64 * 1024 * 1024, governor.clone()));
    GpuLocalParquetSourceCache::register_governor_pressure_observer(&cache).unwrap();
    let file_size = fs::metadata(&path).unwrap().len();

    let build_plan = || {
        let source = LocalParquetSource::from_files(
            vec![LocalParquetFile::new(&path).with_stable_id("stable-input").with_file_size_bytes(file_size)],
            Arc::clone(&source_schema),
            vec!["key".to_string(), "payload".to_string()],
        );

        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(source));
        builder.finish_gpu_resident(source)
    };
    let build_context = || {
        ExecutionContext::builder()
            .with_shared_runtime(Arc::clone(&runtime))
            .with_memory_control_policy(MemoryControlPolicy::Bounded {
                device_budget_bytes: 8 * 1024 * 1024 * 1024,
                max_source_chunk_bytes: 1024 * 1024,
                max_row_groups_per_chunk: 3,
                max_retry_attempts: 1,
                min_retry_chunk_row_groups: 1,
            })
            .with_local_parquet_source_cache(Arc::clone(&cache))
            .build()
            .unwrap()
    };

    let cold = execute_query_plan(&build_context(), &build_plan()).unwrap();
    assert_eq!(cold.metrics().source_cache_inserts, 1);
    drop(cold);
    let retained_bytes = cache.retained_device_bytes().unwrap();
    assert!(retained_bytes > 0);

    let warm = execute_query_plan(&build_context(), &build_plan()).unwrap();
    assert_eq!(warm.metrics().source_cache_hits, 1);
    let QueryOutput::GpuResident(_handle) = warm.output() else {
        panic!("test plan should retain the cached frame as a GPU-resident output");
    };
    assert!(
        governor
            .try_request(GrantRequest::new(100, governor.capacity_bytes(), governor.capacity_bytes()))
            .unwrap()
            .is_none(),
        "pressure admission must not evict a pinned cache entry"
    );
    assert_eq!(cache.retained_device_bytes().unwrap(), retained_bytes);
    assert_eq!(governor.evictable_bytes().unwrap(), retained_bytes);

    drop(warm);
    let grant = governor
        .try_request(GrantRequest::new(101, governor.capacity_bytes(), governor.capacity_bytes()))
        .unwrap()
        .expect("unpinned cache entry should be evicted under pressure");
    assert_eq!(grant.budget_bytes(), governor.capacity_bytes());
    assert_eq!(cache.retained_device_bytes().unwrap(), 0);
    assert_eq!(governor.evictable_bytes().unwrap(), 0);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_local_parquet_source_cache_source_adjacent_aggregate_warm_hit_skips_decode_and_reinsert() {
    let dir = unique_temp_dir("nexus-query-engine-bounded-source-cache-scheduled-aggregate");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_final_merge_retry_input_parquet(&path, 4);
    let output_schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, false)]));
    let governor = MemoryGovernor::new(runtime.device().ordinal(), 64 * 1024 * 1024).unwrap();
    let cache = Arc::new(GpuLocalParquetSourceCache::new_governor_backed(64 * 1024 * 1024, governor));
    let file_size = fs::metadata(&path).unwrap().len();

    let build_plan = || {
        let source = LocalParquetSource::from_files(
            vec![LocalParquetFile::new(&path).with_file_size_bytes(file_size)],
            Arc::clone(&input_schema),
            vec!["key".to_string()],
        );
        let aggregate = AggregateSpec::new(vec![Expr::column("key")], vec![]);
        let sort = SortSpec::new(vec![SortExpr::new(Expr::column("key"), false, false)]);

        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(source));
        let filter = builder.add_filter(
            source,
            Expr::binary(Expr::column("key"), BinaryOp::GtEq, Expr::literal(LiteralValue::Int32(2))),
            Arc::clone(&input_schema),
        );
        let aggregate = builder.add_aggregate(filter, aggregate, Arc::clone(&output_schema));
        let sort = builder.add_sort(aggregate, sort, Arc::clone(&output_schema));
        builder.finish_with_host_sink(sort)
    };
    let build_context = || {
        ExecutionContext::builder()
            .with_shared_runtime(Arc::clone(&runtime))
            .with_memory_control_policy(MemoryControlPolicy::Bounded {
                device_budget_bytes: 8 * 1024 * 1024 * 1024,
                max_source_chunk_bytes: 1024 * 1024,
                max_row_groups_per_chunk: 1,
                max_retry_attempts: 1,
                min_retry_chunk_row_groups: 1,
            })
            .with_source_scheduler_policy(SourceSchedulerPolicy::new(2))
            .with_local_parquet_source_cache(Arc::clone(&cache))
            .build()
            .unwrap()
    };

    let cold = execute_query_plan(&build_context(), &build_plan()).unwrap();
    assert_eq!(cold.metrics().source_cache_hits, 0);
    assert!(cold.metrics().source_cache_inserts > 0);

    let warm = execute_query_plan(&build_context(), &build_plan()).unwrap();
    let warm_metrics = warm.metrics();
    assert!(warm_metrics.source_cache_hits > 0);
    assert_eq!(warm_metrics.source_cache_misses, 0);
    assert_eq!(warm_metrics.source_cache_inserts, 0);
    assert_eq!(warm_metrics.source_cache_insert_skipped, 0);

    let QueryOutput::HostArrow(batch) = warm.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[2, 3]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_local_parquet_source_cache_q06_like_global_aggregate_reuses_all_source_chunks() {
    let dir = unique_temp_dir("nexus-query-engine-bounded-source-cache-q06-like-global-aggregate");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_key_payload_row_grouped_parquet(&path, 1);
    let projected_schema = Arc::new(Schema::new(vec![Field::new("discounted_payload", DataType::Int64, false)]));
    let output_schema = Arc::new(Schema::new(vec![Field::new("revenue", DataType::Int64, true)]));
    let governor = MemoryGovernor::new(runtime.device().ordinal(), 64 * 1024 * 1024).unwrap();
    let cache = Arc::new(GpuLocalParquetSourceCache::new_governor_backed(64 * 1024 * 1024, governor));
    let file_size = fs::metadata(&path).unwrap().len();

    let build_plan = || {
        let source = LocalParquetSource::from_files(
            vec![LocalParquetFile::new(&path).with_file_size_bytes(file_size)],
            Arc::clone(&input_schema),
            vec!["key".to_string(), "payload".to_string()],
        );
        let aggregate = AggregateSpec::new(
            vec![],
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("discounted_payload"), "revenue")],
        );

        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(source));
        let filter = builder.add_filter(
            source,
            Expr::binary(Expr::column("key"), BinaryOp::LtEq, Expr::literal(LiteralValue::Int32(2))),
            Arc::clone(&input_schema),
        );
        let projection = builder.add_projection(
            filter,
            projection_outputs(
                vec![
                    Expr::binary(Expr::column("payload"), BinaryOp::Multiply, Expr::literal(LiteralValue::Int64(2)))
                        .alias("discounted_payload"),
                ],
                Arc::clone(&projected_schema),
            ),
        );
        let aggregate = builder.add_aggregate(projection, aggregate, Arc::clone(&output_schema));
        builder.finish_with_host_sink(aggregate)
    };
    let build_context = || {
        ExecutionContext::builder()
            .with_shared_runtime(Arc::clone(&runtime))
            .with_memory_control_policy(MemoryControlPolicy::Bounded {
                device_budget_bytes: 8 * 1024 * 1024 * 1024,
                max_source_chunk_bytes: 1024 * 1024,
                max_row_groups_per_chunk: 1,
                max_retry_attempts: 1,
                min_retry_chunk_row_groups: 1,
            })
            .with_source_scheduler_policy(SourceSchedulerPolicy::new(2))
            .with_local_parquet_source_cache(Arc::clone(&cache))
            .build()
            .unwrap()
    };

    let cold = execute_query_plan(&build_context(), &build_plan()).unwrap();
    let cold_metrics = cold.metrics();
    assert_eq!(cold_metrics.source_cache_hits, 0);
    assert_eq!(cold_metrics.source_cache_misses, 4);
    assert_eq!(cold_metrics.source_cache_inserts, 4);
    assert_eq!(cold_metrics.source_chunks_committed, 4);

    let warm = execute_query_plan(&build_context(), &build_plan()).unwrap();
    let warm_metrics = warm.metrics();
    assert_eq!(warm_metrics.source_cache_hits, 4);
    assert_eq!(warm_metrics.source_cache_misses, 0);
    assert_eq!(warm_metrics.source_cache_inserts, 0);
    assert_eq!(
        warm_metrics.source_cache_insert_skipped, 4,
        "scheduled warm hits may re-attempt admission, but must resolve as already present without decoding"
    );
    assert_eq!(warm_metrics.source_chunks_committed, 4);

    let QueryOutput::HostArrow(batch) = warm.output() else {
        panic!("host sink should materialize the q06-like global aggregate");
    };
    assert_eq!(batch.num_rows(), 1);
    assert_eq!(int64_array(batch, 0).values().as_ref(), &[70]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_local_parquet_source_cache_binds_execution_frame_schema_with_pure_filter_columns() {
    let dir = unique_temp_dir("nexus-query-engine-bounded-source-cache-execution-frame-schema");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    write_row_grouped_input_parquet(&path);
    let source_schema = Arc::new(Schema::new(vec![
        Field::new("payload", DataType::Int64, false),
        Field::new("key", DataType::Int32, false),
    ]));
    let output_schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, false)]));
    let cache = Arc::new(GpuLocalParquetSourceCache::new(64 * 1024 * 1024));
    let file_size = fs::metadata(&path).unwrap().len();

    let build_plan = || {
        let source = LocalParquetSource::from_files(
            vec![LocalParquetFile::new(&path).with_stable_id("stable-input").with_file_size_bytes(file_size)],
            Arc::clone(&source_schema),
            vec!["payload".to_string(), "key".to_string()],
        )
        .with_column_roles(vec!["key".to_string()], vec!["payload".to_string()]);

        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(source));
        let filter = builder.add_filter(
            source,
            Expr::binary(Expr::column("payload"), BinaryOp::GtEq, Expr::literal(LiteralValue::Int64(20))),
            Arc::clone(&source_schema),
        );
        let projection =
            builder.add_projection(filter, projection_outputs(vec![Expr::column("key")], Arc::clone(&output_schema)));
        builder.finish_with_host_sink(projection)
    };
    let build_context = || {
        ExecutionContext::builder()
            .with_shared_runtime(Arc::clone(&runtime))
            .with_memory_control_policy(MemoryControlPolicy::Bounded {
                device_budget_bytes: 8 * 1024 * 1024 * 1024,
                max_source_chunk_bytes: 1024 * 1024,
                max_row_groups_per_chunk: 1,
                max_retry_attempts: 1,
                min_retry_chunk_row_groups: 1,
            })
            .with_local_parquet_source_cache(Arc::clone(&cache))
            .build()
            .unwrap()
    };

    let first = execute_query_plan(&build_context(), &build_plan()).unwrap();
    assert_eq!(first.metrics().source_cache_misses, 3);
    assert_eq!(first.metrics().source_cache_inserts, 3);

    let second = execute_query_plan(&build_context(), &build_plan()).unwrap();
    let second_metrics = second.metrics();
    assert_eq!(second_metrics.source_cache_hits, 3);
    assert_eq!(second_metrics.source_cache_misses, 0);
    assert!(
        second_metrics.reservation_events.iter().any(|event| {
            event.memory_boundary == "source_cache_borrowed"
                && event.action == MemoryReservationEventAction::AliasAcquire
        }),
        "cache-aware source-adjacent streaming should borrow retained raw chunks before filtering"
    );
    assert!(
        second_metrics.reservation_events.iter().any(|event| {
            event.memory_boundary == "bounded_generic_source_filter_output"
                && event.action == MemoryReservationEventAction::Transfer
        }),
        "cache-aware source-adjacent filtering should transfer ownership to the filtered chunk boundary"
    );
    assert!(
        second_metrics.reservation_events.iter().any(|event| {
            event.memory_boundary == "bounded_generic_source_projection_output"
                && event.action == MemoryReservationEventAction::Transfer
        }),
        "cache-aware source-adjacent projection should transfer ownership to the projected chunk boundary"
    );
    assert!(
        second_metrics.reservation_events.iter().all(|event| event.memory_boundary != "bounded_generic_filter"),
        "cache-aware source-adjacent streaming should not materialize the cached raw source before filtering"
    );

    let QueryOutput::HostArrow(batch) = second.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 3);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[1, 3, 2]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_local_parquet_source_cache_retained_bytes_are_admitted_before_cache_hits() {
    let dir = unique_temp_dir("nexus-query-engine-bounded-source-cache-admission");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let _input_schema = write_row_grouped_input_parquet(&path);
    let source_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int64, false),
    ]));
    let cache = Arc::new(GpuLocalParquetSourceCache::new(64 * 1024 * 1024));
    let file_size = fs::metadata(&path).unwrap().len();

    let build_plan = || {
        let source = LocalParquetSource::from_files(
            vec![LocalParquetFile::new(&path).with_stable_id("stable-input").with_file_size_bytes(file_size)],
            Arc::clone(&source_schema),
            vec!["key".to_string(), "payload".to_string()],
        );

        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(source));
        builder.finish_with_host_sink(source)
    };
    let warm_context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 8 * 1024 * 1024 * 1024,
            max_source_chunk_bytes: 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .with_local_parquet_source_cache(Arc::clone(&cache))
        .build()
        .unwrap();
    execute_query_plan(&warm_context, &build_plan()).unwrap();
    let retained_bytes = cache.retained_device_bytes().unwrap();
    assert!(retained_bytes > 0);

    let constrained_context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: retained_bytes.saturating_sub(1),
            max_source_chunk_bytes: 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .with_local_parquet_source_cache(Arc::clone(&cache))
        .build()
        .unwrap();
    let error = match execute_query_plan(&constrained_context, &build_plan()) {
        Ok(_) => panic!("source-cache retained bytes must be reserved before cache hits can be reused"),
        Err(error) => error,
    };
    assert_eq!(error.code(), QueryErrorCode::MemoryReservationFailed);
    assert_eq!(error.fact_value("memory_boundary"), Some("source_cache_retained"));
    assert_eq!(error.fact_value("reservation_kind"), Some("source_cache_retained"));
    let requested_bytes = error.fact_value("requested_bytes").and_then(|value| value.parse::<u64>().ok());
    assert_eq!(requested_bytes, Some(retained_bytes));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_local_parquet_source_cache_reclaims_retained_bytes_under_reservation_pressure() {
    let dir = unique_temp_dir("nexus-query-engine-bounded-source-cache-pressure");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let _input_schema = write_row_grouped_input_parquet(&path);
    let source_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int64, false),
    ]));
    let cache = Arc::new(GpuLocalParquetSourceCache::new(64 * 1024 * 1024));
    let file_size = fs::metadata(&path).unwrap().len();

    let build_plan = |source_path: &std::path::Path, stable_id: &str| {
        let source = LocalParquetSource::from_files(
            vec![
                LocalParquetFile::new(source_path)
                    .with_stable_id(stable_id)
                    .with_file_size_bytes(file_size)
                    .with_row_group_bytes(vec![8192, 8192, 8192])
                    .with_projected_row_group_bytes(vec![8192, 8192, 8192], vec![8192, 8192, 8192]),
            ],
            Arc::clone(&source_schema),
            vec!["key".to_string(), "payload".to_string()],
        );

        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(source));
        builder.finish_gpu_resident(source)
    };
    let build_context = |device_budget_bytes, max_row_groups_per_chunk| {
        ExecutionContext::builder()
            .with_shared_runtime(Arc::clone(&runtime))
            .with_memory_control_policy(MemoryControlPolicy::Bounded {
                device_budget_bytes,
                max_source_chunk_bytes: 1024 * 1024,
                max_row_groups_per_chunk,
                max_retry_attempts: 1,
                min_retry_chunk_row_groups: 1,
            })
            .with_local_parquet_source_cache(Arc::clone(&cache))
            .build()
            .unwrap()
    };

    let mut max_source_read_request = None;
    for index in 0..3 {
        let warm_path = dir.join(format!("warm-{index}.parquet"));
        fs::copy(&path, &warm_path).unwrap();
        let stable_id = format!("stable-input-a{index}");
        let warm =
            execute_query_plan(&build_context(8 * 1024 * 1024 * 1024, 3), &build_plan(&warm_path, &stable_id)).unwrap();
        assert_eq!(warm.metrics().source_cache_inserts, 1);
        max_source_read_request = warm
            .metrics()
            .reservation_events
            .iter()
            .filter(|event| event.reservation_kind == ReservationKind::SourceRead)
            .map(|event| event.requested_bytes)
            .max();
    }
    let retained_bytes = cache.retained_device_bytes().unwrap();
    assert!(retained_bytes > 0);
    let max_source_read_request =
        max_source_read_request.expect("warm source read should record a reservation request");

    let pressured = execute_query_plan(
        &build_context(retained_bytes.saturating_add(max_source_read_request).saturating_sub(1), 3),
        &build_plan(&path, "stable-input-b"),
    )
    .unwrap();
    assert_eq!(pressured.metrics().source_cache_hits, 0);
    assert_eq!(pressured.metrics().source_cache_misses, 1);
    assert!(pressured.metrics().source_cache_inserts > 0);
    assert_eq!(pressured.metrics().reservation_denied_count, 0);

    let QueryOutput::GpuResident(handle) = pressured.output() else {
        panic!("gpu sink should retain one GPU result handle");
    };
    assert_eq!(handle.num_rows(), 6);
    assert_eq!(handle.num_columns(), 2);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_query_executes_group_only_aggregate_as_distinct_keys() {
    let dir = unique_temp_dir("nexus-query-engine-native-group-only-aggregate");
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

    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime)).unwrap();
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
    let dir = unique_temp_dir("nexus-query-engine-bounded-group-only-aggregate");
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
fn native_query_executes_group_only_aggregate_when_output_key_name_already_exists() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let schema = Arc::new(Schema::new(vec![
        Field::new("i_item_id", DataType::Utf8, false),
        Field::new("__cudf_group_expr_0_i_item_id", DataType::Utf8, false),
        Field::new("i_item_desc", DataType::Utf8, false),
        Field::new("i_current_price", DataType::Int32, false),
    ]));
    let input_batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(vec!["carry_a", "carry_b", "carry_c", "carry_d"])),
            Arc::new(StringArray::from(vec!["item_1", "item_1", "item_2", "item_2"])),
            Arc::new(StringArray::from(vec!["desc_1", "desc_1", "desc_2", "desc_2"])),
            Arc::new(Int32Array::from(vec![10, 10, 20, 20])),
        ],
    )
    .unwrap();
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("i_item_id", DataType::Utf8, false),
        Field::new("i_item_desc", DataType::Utf8, false),
        Field::new("i_current_price", DataType::Int32, false),
    ]));
    let aggregate = AggregateSpec::new(
        vec![
            Expr::column("__cudf_group_expr_0_i_item_id").alias("i_item_id"),
            Expr::column("i_item_desc"),
            Expr::column("i_current_price"),
        ],
        vec![],
    );
    let sort = SortSpec::new(vec![SortExpr::new(Expr::column("i_item_id"), false, false)]);

    let mut builder = QueryPlanBuilder::new();
    let source =
        builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(Arc::clone(&schema), vec![input_batch])));
    let aggregate = builder.add_aggregate(source, aggregate, Arc::clone(&output_schema));
    let sort = builder.add_sort(aggregate, sort, Arc::clone(&output_schema));
    let plan = builder.finish_with_host_sink(sort);

    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime)).unwrap();
    let result = execute_query_plan(&context, &plan).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 2);
    assert_eq!(batch.num_columns(), 3);
    assert_eq!(string_array(batch, 0).iter().collect::<Vec<_>>(), vec![Some("item_1"), Some("item_2")]);
    assert_eq!(string_array(batch, 1).iter().collect::<Vec<_>>(), vec![Some("desc_1"), Some("desc_2")]);
    assert_eq!(int32_array(batch, 2).values().as_ref(), &[10, 20]);
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
