#[test]
fn native_local_parquet_source_carries_timestamp_timezone_to_host_output() {
    let dir = unique_temp_dir("nexus-query-engine-native-timestamp-tz");
    let path = dir.join("input.parquet");
    let context = backend_context();
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
    let context = backend_context();
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
    let context = backend_context();
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
