use super::*;

#[test]
fn native_query_preserves_multi_file_source_order() {
    let dir = unique_temp_dir("nexus-query-engine-multi-file-source-order");
    let first_path = dir.join("part-000.parquet");
    let second_path = dir.join("part-001.parquet");
    let third_path = dir.join("part-002.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let schema = write_i32_parquet_with_row_group_rows(&first_path, &[1, 2], 1);
    write_i32_parquet_with_row_group_rows(&second_path, &[10, 11], 1);
    write_i32_parquet_with_row_group_rows(&third_path, &[20], 1);
    let source = LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new(&second_path)
                .with_file_size_bytes(fs::metadata(&second_path).unwrap().len())
                .with_row_group_rows(vec![1, 1]),
            LocalParquetFile::new(&first_path)
                .with_file_size_bytes(fs::metadata(&first_path).unwrap().len())
                .with_row_group_rows(vec![1, 1]),
            LocalParquetFile::new(&third_path)
                .with_file_size_bytes(fs::metadata(&third_path).unwrap().len())
                .with_row_group_rows(vec![1]),
        ],
        Arc::clone(&schema),
        vec!["id".to_string()],
    );

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let plan = builder.finish_with_host_sink(source);
    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 1024 * 1024,
            max_source_chunk_bytes: 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .build()
        .unwrap();

    let result = execute_query_plan(&context, &plan).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should return a RecordBatch");
    };
    assert_eq!(batch.num_rows(), 5);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[10, 11, 1, 2, 20]);
    assert_eq!(result.metrics().parquet_files, 3);
    assert_eq!(result.metrics().source_chunks_committed, 5);

    fs::remove_dir_all(dir).unwrap();
}
