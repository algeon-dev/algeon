use std::sync::Arc;

use super::*;

#[test]
fn bounded_native_query_executes_non_aggregate_source_plan() {
    let dir = unique_temp_dir("query-engine-bounded-source-only");
    let path = dir.join("input.parquet");
    let schema = write_row_grouped_input_parquet(&path);
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&schema),
        vec!["key".into(), "payload".into(), "m00".into(), "m01".into()],
    );

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let plan = builder.finish_with_host_sink(source);
    let context = backend_context_with_grant(1024 * 1024);

    let result = execute_query_plan(&context, &plan).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should return a RecordBatch");
    };
    assert_eq!(batch.num_rows(), 6);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[1, 2, 1, 3, 2, 3]);
    assert_eq!(int64_array(batch, 1).values().as_ref(), &[10, 10, 20, 20, 30, 5]);
    assert_eq!(result.metrics().parquet_files, 1);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_preserves_multi_file_source_order() {
    let dir = unique_temp_dir("query-engine-bounded-multi-file-source-order");
    let first_path = dir.join("part-000.parquet");
    let second_path = dir.join("part-001.parquet");
    let third_path = dir.join("part-002.parquet");
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
    let context = backend_context_with_grant(1024 * 1024);

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

#[test]
fn bounded_native_query_executes_union_plan_without_legacy_aggregate_path() {
    let context = backend_context_with_grant(1024 * 1024);
    let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, false)]));
    let left_batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from(vec![1, 2]))]).unwrap();
    let right_batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from(vec![3, 4]))]).unwrap();

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("left", Arc::clone(&schema))));
    let right = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("right", Arc::clone(&schema))));
    let union = builder.add_union(vec![left, right], Arc::clone(&schema));
    let plan = builder.finish_with_host_sink(union);
    let admitted = context.admit_plan(&plan).unwrap();
    let runtime = Arc::new(admitted.runtime().unwrap());
    let mut gpu_inputs = GpuResidentInputMap::new();
    gpu_inputs.insert(
        "left".to_string(),
        GpuResultHandle::new(
            &admitted,
            GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &left_batch).unwrap()),
            Arc::clone(&schema),
        )
        .unwrap(),
    );
    gpu_inputs.insert(
        "right".to_string(),
        GpuResultHandle::new(
            &admitted,
            GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &right_batch).unwrap()),
            Arc::clone(&schema),
        )
        .unwrap(),
    );

    let result = execute_query_plan_with_gpu_inputs(&admitted, &plan, &gpu_inputs).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should return a RecordBatch");
    };
    assert_eq!(batch.num_rows(), 4);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[1, 2, 3, 4]);
    admitted.finish();
}

#[test]
fn native_query_preserves_multi_file_source_order() {
    let dir = unique_temp_dir("query-engine-multi-file-source-order");
    let first_path = dir.join("part-000.parquet");
    let second_path = dir.join("part-001.parquet");
    let third_path = dir.join("part-002.parquet");
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
    let context = backend_context_with_memory_control_policy(MemoryControlPolicy {
        device_budget_bytes: 1024 * 1024,
        max_source_chunk_bytes: 1024,
        max_row_groups_per_chunk: 1,
        max_retry_attempts: 1,
        min_retry_chunk_row_groups: 1,
    });

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
