use super::*;

#[test]
fn native_query_executes_multi_file_direct_parquet_source_in_file_order() {
    let dir = unique_temp_dir("query-engine-native-execution-multi-file-order");
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
    let plan = builder.finish_with_host_sink(source).unwrap();

    let context = backend_context();
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
fn native_context_restores_duplicate_host_sink_schema_without_swapping_columns() {
    let dir = unique_temp_dir("query-engine-unbounded-duplicate-host-sink-schema");
    let path = dir.join("input.parquet");
    let runtime = backend_runtime();
    let input_schema = write_input_parquet(&runtime, &path);
    let executable_schema = Arc::new(Schema::new(vec![
        Field::new("exec_left_value", DataType::Int64, false),
        Field::new("exec_right_value", DataType::Int64, false),
    ]));
    let frontend_sink_schema = Arc::new(Schema::new(vec![
        Field::new("value", DataType::Int64, false),
        Field::new("value", DataType::Int64, false),
    ]));
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["m00".to_string(), "m01".to_string()],
    );

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let projection = builder
        .add_projection(
            source,
            projection_outputs(
                vec![Expr::column("m00").alias("exec_left_value"), Expr::column("m01").alias("exec_right_value")],
                executable_schema,
            ),
        )
        .unwrap();
    let plan = builder.finish_with_host_sink_schema(projection, Arc::clone(&frontend_sink_schema)).unwrap();
    let context = backend_context();

    let result = execute_query_plan(&context, &plan).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.schema().as_ref(), frontend_sink_schema.as_ref());
    assert_eq!(batch.schema().field(0).name(), "value");
    assert_eq!(batch.schema().field(1).name(), "value");
    assert_eq!(int64_array(batch, 0).values().as_ref(), &[100, 200, 300, 400]);
    assert_eq!(int64_array(batch, 1).values().as_ref(), &[1, 2, 3, 4]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_context_executes_direct_parquet_pipeline_to_host_sink() {
    let dir = unique_temp_dir("query-engine-unbounded-native-execution");
    let path = dir.join("input.parquet");
    let runtime = backend_runtime();
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
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
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
    let filter = builder.add_filter(source, filter).unwrap();
    let projection = builder.add_projection(filter, projection_outputs(projection, projected_schema)).unwrap();
    let aggregate = builder.add_aggregate(projection, aggregate, Arc::clone(&aggregated_schema)).unwrap();
    let sort = builder.add_sort(aggregate, sort).unwrap();
    let limit = builder.add_limit(sort, LimitSpec::fetch(2)).unwrap();
    let plan = builder.finish_with_host_sink(limit).unwrap();

    let context = backend_context();
    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.parquet_files, 1);
    assert_eq!(metrics.read_columns, 4);
    assert_eq!(metrics.input_batches, 1);
    assert_eq!(metrics.input_rows, 4);
    assert_eq!(metrics.source_read_runtime_rows.len(), 1);
    let source_read_row = &metrics.source_read_runtime_rows[0];
    assert_eq!(source_read_row.source_node_id, source.to_string());
    assert_eq!(source_read_row.read_kind, "chunk");
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
fn native_context_executes_multi_file_direct_parquet_source() {
    let dir = unique_temp_dir("query-engine-unbounded-native-execution-multi-file");
    let first_path = dir.join("part-000.parquet");
    let second_path = dir.join("part-001.parquet");
    let runtime = backend_runtime();
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
    let filter = builder.add_filter(source, filter).unwrap();
    let projection = builder.add_projection(filter, projection_outputs(projection, projected_schema)).unwrap();
    let aggregate = builder.add_aggregate(projection, aggregate, Arc::clone(&aggregated_schema)).unwrap();
    let sort = builder.add_sort(aggregate, sort).unwrap();
    let limit = builder.add_limit(sort, LimitSpec::fetch(2)).unwrap();
    let plan = builder.finish_with_host_sink(limit).unwrap();

    let context = backend_context();
    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.parquet_files, 2);
    assert_eq!(metrics.read_columns, 8);
    assert_eq!(metrics.input_batches, 1);
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
