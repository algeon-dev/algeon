#[test]
fn native_local_parquet_source_applies_iceberg_position_deletes_before_output_and_count_probe() {
    let dir = unique_temp_dir("nexus-query-engine-native-position-deletes");
    let data_path = dir.join("data.parquet");
    let delete_path = dir.join("delete.parquet");
    let context = backend_context();
    let runtime = Arc::new(context.runtime().unwrap());
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
    let context = backend_context();
    let runtime = Arc::new(context.runtime().unwrap());

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
    let context = backend_context();
    let runtime = Arc::new(context.runtime().unwrap());

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
    let context = backend_context();

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
    let context = backend_context();

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
