fn expect_device_buffer_error(result: cudf::Result<cudf::core::DeviceBuffer>) -> cudf::Error {
    match result {
        Ok(_) => panic!("expected device buffer allocation to fail"),
        Err(error) => error,
    }
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
