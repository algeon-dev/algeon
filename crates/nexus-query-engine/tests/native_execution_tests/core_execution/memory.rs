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
