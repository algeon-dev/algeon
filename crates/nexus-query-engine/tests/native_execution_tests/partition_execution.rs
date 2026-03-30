use super::*;

#[test]
fn native_query_executes_filter_projection_and_group_by_on_partition_columns() {
    let dir = unique_temp_dir("nexus-query-engine-native-execution-partitions");
    let first_path = dir.join("part-001.parquet");
    let second_path = dir.join("part-002.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let file_schema = write_input_parquet(&runtime, &first_path);
    write_input_parquet(&runtime, &second_path);
    let plan = partition_column_query_plan(&first_path, &second_path, file_schema);

    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime)).unwrap();
    let result = execute_query_plan(&context, &plan).unwrap();

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 1);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[2]);
    assert_eq!(int64_array(batch, 1).values().as_ref(), &[1000]);
    assert_eq!(result.metrics().parquet_files, 2);
    assert_eq!(result.metrics().read_columns, 2);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_executes_partition_columns_per_source_chunk() {
    let dir = unique_temp_dir("nexus-query-engine-native-execution-bounded-partitions");
    let first_path = dir.join("part-001.parquet");
    let second_path = dir.join("part-002.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let file_schema = write_input_parquet(&runtime, &first_path);
    write_input_parquet(&runtime, &second_path);
    let plan = partition_column_query_plan(&first_path, &second_path, file_schema);

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

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 1);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[2]);
    assert_eq!(int64_array(batch, 1).values().as_ref(), &[1000]);
    assert_eq!(result.metrics().source_chunks_planned, 2);
    assert_eq!(result.metrics().source_chunks_committed, 2);
    let source_events = budgeted_events(result.metrics(), "nexus_query_engine.local_parquet_chunk.read");
    assert_eq!(source_events.len(), 2);
    for event in source_events {
        assert_source_budget_event_context(event, "m00", "part,m00");
        assert_eq!(event.context_facts.get("reads_all_row_groups").map(String::as_str), Some("false"));
        assert_eq!(event.context_facts.get("selected_row_groups").map(String::as_str), Some("0"));
        assert_eq!(event.context_facts.get("selected_row_group_count").map(String::as_str), Some("1"));
        assert_eq!(event.context_facts.get("read_column_count").map(String::as_str), Some("1"));
        assert_eq!(event.context_facts.get("projected_column_count").map(String::as_str), Some("2"));
        assert_eq!(event.context_facts.get("required_columns").map(String::as_str), Some("part"));
        assert_eq!(event.context_facts.get("partition_columns").map(String::as_str), Some("part"));
    }

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_executes_typed_null_partition_columns_per_source_chunk() {
    let dir = unique_temp_dir("nexus-query-engine-native-execution-bounded-null-partitions");
    let first_path = dir.join("part-001.parquet");
    let second_path = dir.join("part-002.parquet");
    let file_schema = write_single_id_parquet(&first_path, 1);
    write_single_id_parquet(&second_path, 2);
    let partition_columns = vec![
        PartitionColumn::new("null_amount", DataType::Decimal128(15, 2), true),
        PartitionColumn::new("null_shipdate", DataType::Date32, true),
        PartitionColumn::new("null_event_ts", DataType::Timestamp(TimeUnit::Nanosecond, None), true),
        PartitionColumn::new("null_elapsed", DataType::Duration(TimeUnit::Nanosecond), true),
    ];
    let null_values = vec![
        LiteralValue::Null(DataType::Decimal128(15, 2)),
        LiteralValue::Null(DataType::Date32),
        LiteralValue::Null(DataType::Timestamp(TimeUnit::Nanosecond, None)),
        LiteralValue::Null(DataType::Duration(TimeUnit::Nanosecond)),
    ];
    let plan = partition_passthrough_query_plan(
        &[first_path.as_path(), second_path.as_path()],
        file_schema,
        partition_columns,
        vec![null_values.clone(), null_values],
    );
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let context = bounded_source_chunk_context(Arc::clone(&runtime));
    let result = execute_query_plan(&context, &plan).unwrap();

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 2);
    assert_eq!(int64_array(batch, 0).values().as_ref(), &[1, 2]);
    assert!(matches!(decimal128_array(batch, 1).data_type(), DataType::Decimal128(_, 2)));
    assert_eq!(decimal128_array(batch, 1).null_count(), 2);
    assert_eq!(date32_array(batch, 2).data_type(), &DataType::Date32);
    assert_eq!(date32_array(batch, 2).null_count(), 2);
    assert_eq!(timestamp_ns_array(batch, 3).data_type(), &DataType::Timestamp(TimeUnit::Nanosecond, None));
    assert_eq!(timestamp_ns_array(batch, 3).null_count(), 2);
    assert_eq!(duration_ns_array(batch, 4).data_type(), &DataType::Duration(TimeUnit::Nanosecond));
    assert_eq!(duration_ns_array(batch, 4).null_count(), 2);
    assert_eq!(int64_array(batch, 5).values().as_ref(), &[1, 2]);
    assert_eq!(result.metrics().source_chunks_planned, 2);
    assert_eq!(result.metrics().source_chunks_committed, 2);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_executes_extended_partition_columns_per_source_chunk() {
    let dir = unique_temp_dir("nexus-query-engine-native-execution-bounded-extended-partitions");
    let first_path = dir.join("part-001.parquet");
    let second_path = dir.join("part-002.parquet");
    let file_schema = write_single_id_parquet(&first_path, 1);
    write_single_id_parquet(&second_path, 2);
    let first_date = date32(2023, 5, 4);
    let second_date = date32(2023, 5, 5);
    let partition_columns = vec![
        PartitionColumn::new("amount", DataType::Decimal128(15, 2), false),
        PartitionColumn::new("shipdate", DataType::Date32, false),
        PartitionColumn::new("event_ts", DataType::Timestamp(TimeUnit::Nanosecond, None), false),
        PartitionColumn::new("elapsed", DataType::Duration(TimeUnit::Nanosecond), false),
    ];
    let plan = partition_passthrough_query_plan(
        &[first_path.as_path(), second_path.as_path()],
        file_schema,
        partition_columns,
        vec![
            vec![
                LiteralValue::Decimal128 { raw: 12_345, precision: 15, scale: 2 },
                LiteralValue::Date32(first_date),
                LiteralValue::Timestamp { raw: 1_700_000_123_456_789_000, unit: TimeUnit::Nanosecond },
                LiteralValue::Duration { raw: -42_000_000, unit: TimeUnit::Nanosecond },
            ],
            vec![
                LiteralValue::Decimal128 { raw: 67_890, precision: 15, scale: 2 },
                LiteralValue::Date32(second_date),
                LiteralValue::Timestamp { raw: 1_700_000_223_456_789_000, unit: TimeUnit::Nanosecond },
                LiteralValue::Duration { raw: 17_000, unit: TimeUnit::Nanosecond },
            ],
        ],
    );
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let context = bounded_source_chunk_context(Arc::clone(&runtime));
    let result = execute_query_plan(&context, &plan).unwrap();

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 2);
    assert_eq!(int64_array(batch, 0).values().as_ref(), &[1, 2]);
    let amount = decimal128_array(batch, 1);
    assert!(matches!(amount.data_type(), DataType::Decimal128(_, 2)));
    assert_eq!(amount.values().as_ref(), &[12_345, 67_890]);
    assert_eq!(date32_array(batch, 2).data_type(), &DataType::Date32);
    assert_eq!(date32_array(batch, 2).values().as_ref(), &[first_date, second_date]);
    assert_eq!(timestamp_ns_array(batch, 3).data_type(), &DataType::Timestamp(TimeUnit::Nanosecond, None));
    assert_eq!(timestamp_ns_array(batch, 3).values().as_ref(), &[1_700_000_123_456_789_000, 1_700_000_223_456_789_000]);
    assert_eq!(duration_ns_array(batch, 4).data_type(), &DataType::Duration(TimeUnit::Nanosecond));
    assert_eq!(duration_ns_array(batch, 4).values().as_ref(), &[-42_000_000, 17_000]);
    assert_eq!(int64_array(batch, 5).values().as_ref(), &[1, 2]);
    assert_eq!(result.metrics().source_chunks_planned, 2);
    assert_eq!(result.metrics().source_chunks_committed, 2);

    fs::remove_dir_all(dir).unwrap();
}
