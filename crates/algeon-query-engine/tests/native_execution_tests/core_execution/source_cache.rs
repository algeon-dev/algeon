#[test]
fn native_local_parquet_source_scan_limit_applies_before_parent_operators() {
    let dir = unique_temp_dir("query-engine-native-source-scan-limit");
    let path = dir.join("input.parquet");
    let context = backend_context();
    let runtime = Arc::new(context.runtime().unwrap());
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
fn native_local_parquet_source_materializes_missing_nullable_carry_columns() {
    let dir = unique_temp_dir("query-engine-native-missing-nullable-source-columns");
    let path = dir.join("input.parquet");
    let context = backend_context();
    let runtime = Arc::new(context.runtime().unwrap());
    write_input_parquet(&runtime, &path);

    let list_type = DataType::List(Arc::new(Field::new("item", DataType::Utf8, true)));
    let large_float_list_type = DataType::LargeList(Arc::new(Field::new("item", DataType::Float64, true)));
    let fixed_integer_list_type = DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Int32, true)), 2);
    let decimal32_type = DataType::Decimal32(8, 2);
    let decimal64_type = DataType::Decimal64(17, 4);
    let source_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("missing_tags", list_type.clone(), true),
        Field::new("missing_scores", large_float_list_type.clone(), true),
        Field::new("missing_coordinates", fixed_integer_list_type.clone(), true),
        Field::new("missing_amount32", decimal32_type.clone(), true),
        Field::new("missing_amount64", decimal64_type.clone(), true),
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
    .with_missing_columns(vec![
        MissingColumn::new("missing_tags", list_type, true),
        MissingColumn::new("missing_scores", large_float_list_type, true),
        MissingColumn::new("missing_coordinates", fixed_integer_list_type, true),
        MissingColumn::new("missing_amount32", decimal32_type, true),
        MissingColumn::new("missing_amount64", decimal64_type, true),
    ])
    .with_column_roles(
        vec![
            "key".to_string(),
            "missing_tags".to_string(),
            "missing_scores".to_string(),
            "missing_coordinates".to_string(),
            "missing_amount32".to_string(),
            "missing_amount64".to_string(),
        ],
        Vec::new(),
    );

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let plan = builder.finish_with_host_sink(source);

    let result = execute_query_plan(&context, &plan).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 4);
    assert_eq!(batch.num_columns(), 6);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[1, 2, 1, 3]);
    assert_eq!(batch.schema(), source_schema);
    let missing = batch.column(1).as_any().downcast_ref::<ListArray>().unwrap();
    assert_eq!(missing.null_count(), 4);
    assert_eq!(missing.values().len(), 0);
    let missing_scores = batch.column(2).as_any().downcast_ref::<LargeListArray>().unwrap();
    assert_eq!(missing_scores.null_count(), 4);
    assert_eq!(missing_scores.values().len(), 0);
    let missing_coordinates = batch.column(3).as_any().downcast_ref::<FixedSizeListArray>().unwrap();
    assert_eq!(missing_coordinates.null_count(), 4);
    assert_eq!(missing_coordinates.value_length(), 2);
    assert_eq!(missing_coordinates.values().len(), 8);
    assert_eq!(batch.column(4).data_type(), &DataType::Decimal32(8, 2));
    assert_eq!(batch.column(4).null_count(), 4);
    assert_eq!(batch.column(5).data_type(), &DataType::Decimal64(17, 4));
    assert_eq!(batch.column(5).null_count(), 4);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_local_parquet_source_materializes_missing_nullable_boolean_lists() {
    let dir = unique_temp_dir("query-engine-bounded-missing-nullable-source-columns");
    let path = dir.join("input.parquet");
    let context = backend_context();
    let runtime = Arc::new(context.runtime().unwrap());
    write_input_parquet(&runtime, &path);

    let list_type = DataType::List(Arc::new(Field::new("item", DataType::Boolean, true)));
    let source_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("missing_flags", list_type.clone(), true),
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
    .with_missing_columns(vec![MissingColumn::new("missing_flags", list_type, true)])
    .with_column_roles(vec!["key".to_string(), "missing_flags".to_string()], Vec::new());

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let plan = builder.finish_with_host_sink(source);

    let result = execute_query_plan(&context, &plan).unwrap();
    assert_eq!(result.metrics().source_chunks_committed, 1);
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 4);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[1, 2, 1, 3]);
    assert_eq!(batch.schema(), source_schema);
    let missing = batch.column(1).as_any().downcast_ref::<ListArray>().unwrap();
    assert_eq!(missing.null_count(), 4);
    assert_eq!(missing.values().len(), 0);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_zero_read_column_source_materializes_missing_integer_lists_for_nonzero_and_zero_rows() {
    let dir = unique_temp_dir("query-engine-native-zero-read-missing-list");
    let path = dir.join("input.parquet");
    let context = backend_context();
    let runtime = Arc::new(context.runtime().unwrap());
    write_input_parquet(&runtime, &path);

    let list_type = DataType::List(Arc::new(Field::new("item", DataType::Int64, false)));
    let source_schema = Arc::new(Schema::new(vec![Field::new("missing_ports", list_type.clone(), true)]));
    for (scan_limit, expected_rows) in [(None, 4), (Some(0), 0)] {
        let source = LocalParquetSource::from_files(
            vec![
                LocalParquetFile::new(&path)
                    .with_file_size_bytes(fs::metadata(&path).unwrap().len())
                    .with_row_group_rows(vec![4]),
            ],
            Arc::clone(&source_schema),
            Vec::new(),
        )
        .with_missing_columns(vec![MissingColumn::new("missing_ports", list_type.clone(), true)])
        .with_column_roles(vec!["missing_ports".to_string()], Vec::new())
        .with_scan_limit(scan_limit);
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(source));
        let plan = builder.finish_with_host_sink(source);

        let result = execute_query_plan(&context, &plan).unwrap();
        let QueryOutput::HostArrow(batch) = result.output() else {
            panic!("host sink should materialize one RecordBatch");
        };
        assert_eq!(batch.schema(), source_schema);
        assert_eq!(batch.num_rows(), expected_rows);
        let missing = batch.column(0).as_any().downcast_ref::<ListArray>().unwrap();
        assert_eq!(missing.null_count(), expected_rows);
        assert_eq!(missing.values().len(), 0);
    }

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_generic_local_parquet_pipeline_uses_required_only_source_scheduler() {
    let dir = unique_temp_dir("query-engine-bounded-generic-source-scheduler");
    let path = dir.join("input.parquet");
    let context = backend_context();
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

    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.source_chunks_planned, 3);
    assert_eq!(metrics.source_chunks_attempted, 3);
    assert_eq!(metrics.source_chunks_committed, 3);
    assert_eq!(metrics.source_read_prefetches_started, 0);
    assert_eq!(metrics.source_read_prefetches_completed, 0);
    assert_eq!(metrics.source_read_prefetches_skipped, 0);
    assert_eq!(metrics.source_read_prefetches_abandoned, 0);

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
    let dir = unique_temp_dir("query-engine-bounded-generic-source-batch");
    let paths = [dir.join("input-0.parquet"), dir.join("input-1.parquet"), dir.join("input-2.parquet")];
    let context = backend_context();
    let runtime = Arc::new(context.runtime().unwrap());
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
    let dir = unique_temp_dir("query-engine-bounded-source-scan-limit");
    let path = dir.join("input.parquet");
    let context = backend_context_with_memory_and_scheduler_policy(
        MemoryControlPolicy::Bounded {
            device_budget_bytes: 8 * 1024 * 1024 * 1024,
            max_source_chunk_bytes: 8 * 1024 * 1024 * 1024,
            max_row_groups_per_chunk: 8,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        },
        SourceSchedulerPolicy::new(3),
    );
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
