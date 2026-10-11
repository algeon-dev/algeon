use super::*;

fn basic_list_child_array(data_type: &DataType) -> ArrayRef {
    macro_rules! integer_array {
        ($array:ty, $native:ty) => {
            Arc::new(<$array>::from(vec![Some(1 as $native), None, Some(3 as $native)]))
        };
    }
    match data_type {
        DataType::Boolean => Arc::new(BooleanArray::from(vec![Some(true), None, Some(false)])),
        DataType::Int8 => integer_array!(Int8Array, i8),
        DataType::Int16 => integer_array!(Int16Array, i16),
        DataType::Int32 => integer_array!(Int32Array, i32),
        DataType::Int64 => integer_array!(Int64Array, i64),
        DataType::UInt8 => integer_array!(UInt8Array, u8),
        DataType::UInt16 => integer_array!(UInt16Array, u16),
        DataType::UInt32 => integer_array!(UInt32Array, u32),
        DataType::UInt64 => integer_array!(UInt64Array, u64),
        DataType::Float32 => Arc::new(Float32Array::from(vec![Some(f32::NAN), None, Some(f32::NEG_INFINITY)])),
        DataType::Float64 => Arc::new(Float64Array::from(vec![Some(-0.0), None, Some(f64::INFINITY)])),
        DataType::Utf8 => Arc::new(StringArray::from(vec![Some("one"), None, Some("three")])),
        DataType::LargeUtf8 => Arc::new(LargeStringArray::from(vec![Some("one"), None, Some("three")])),
        DataType::Utf8View => Arc::new(StringViewArray::from(vec![Some("one"), None, Some("three")])),
        data_type => panic!("unsupported basic list child fixture type: {data_type:?}"),
    }
}

fn basic_list_record_batch() -> RecordBatch {
    let child_types = [
        ("boolean", DataType::Boolean),
        ("int8", DataType::Int8),
        ("int16", DataType::Int16),
        ("int32", DataType::Int32),
        ("int64", DataType::Int64),
        ("uint8", DataType::UInt8),
        ("uint16", DataType::UInt16),
        ("uint32", DataType::UInt32),
        ("uint64", DataType::UInt64),
        ("float32", DataType::Float32),
        ("float64", DataType::Float64),
        ("utf8", DataType::Utf8),
        ("large_utf8", DataType::LargeUtf8),
        ("utf8_view", DataType::Utf8View),
    ];
    let mut fields = Vec::new();
    let mut columns = Vec::new();
    for (name, child_type) in child_types {
        let child = Arc::new(Field::new("element", child_type.clone(), true));
        let list = ListArray::new(
            Arc::clone(&child),
            OffsetBuffer::new(ScalarBuffer::from(vec![0, 2, 2, 3])),
            basic_list_child_array(&child_type),
            Some(NullBuffer::from(vec![true, false, true])),
        );
        fields.push(Field::new(format!("list_{name}"), DataType::List(child), true));
        columns.push(Arc::new(list) as ArrayRef);
    }
    for (name, child_type) in [
        ("int64", DataType::Int64),
        ("boolean", DataType::Boolean),
        ("float32", DataType::Float32),
        ("float64", DataType::Float64),
        ("utf8", DataType::Utf8),
        ("large_utf8", DataType::LargeUtf8),
        ("utf8_view", DataType::Utf8View),
    ] {
        let child = Arc::new(Field::new("element", child_type.clone(), true));
        let list = LargeListArray::new(
            Arc::clone(&child),
            OffsetBuffer::new(ScalarBuffer::from(vec![0i64, 2, 2, 3])),
            basic_list_child_array(&child_type),
            Some(NullBuffer::from(vec![true, false, true])),
        );
        fields.push(Field::new(format!("large_list_{name}"), DataType::LargeList(child), true));
        columns.push(Arc::new(list) as ArrayRef);
    }
    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap()
}

fn assert_record_batch_columns_equal(actual: &RecordBatch, expected: &RecordBatch) {
    assert_eq!(actual.schema(), expected.schema());
    assert_eq!(actual.num_rows(), expected.num_rows());
    for (index, (actual, expected)) in actual.columns().iter().zip(expected.columns()).enumerate() {
        assert_eq!(actual.to_data(), expected.to_data(), "column {index}");
    }
}

pub(crate) fn basic_list_payload_batch(keys: &[i32]) -> RecordBatch {
    let offsets = OffsetBuffer::new(ScalarBuffer::from((0..=keys.len() as i32).collect::<Vec<_>>()));
    let int_child = Arc::new(Field::new("element", DataType::Int64, true));
    let bool_child = Arc::new(Field::new("element", DataType::Boolean, true));
    let string_child = Arc::new(Field::new("element", DataType::Utf8, true));
    let float_child = Arc::new(Field::new("element", DataType::Float32, true));
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("discarded", DataType::Int64, false),
        Field::new("ints", DataType::List(Arc::clone(&int_child)), true),
        Field::new("flags", DataType::List(Arc::clone(&bool_child)), true),
        Field::new("tags", DataType::List(Arc::clone(&string_child)), true),
        Field::new("scores", DataType::List(Arc::clone(&float_child)), true),
    ]));
    let tags = keys.iter().map(|key| Some(format!("tag-{key}"))).collect::<Vec<_>>();
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(keys.to_vec())),
            Arc::new(Int64Array::from(keys.iter().map(|key| i64::from(*key) * 100).collect::<Vec<_>>())),
            Arc::new(ListArray::new(
                int_child,
                offsets.clone(),
                Arc::new(Int64Array::from(keys.iter().map(|key| i64::from(*key) * 10).collect::<Vec<_>>())),
                None,
            )),
            Arc::new(ListArray::new(
                bool_child,
                offsets.clone(),
                Arc::new(BooleanArray::from(keys.iter().map(|key| key % 2 == 0).collect::<Vec<_>>())),
                None,
            )),
            Arc::new(ListArray::new(string_child, offsets, Arc::new(StringArray::from(tags)), None)),
            Arc::new(ListArray::new(
                float_child,
                OffsetBuffer::new(ScalarBuffer::from((0..=keys.len() as i32).collect::<Vec<_>>())),
                Arc::new(Float32Array::from(keys.iter().map(|key| *key as f32 + 0.25).collect::<Vec<_>>())),
                None,
            )),
        ],
    )
    .unwrap()
}

pub(crate) fn assert_basic_list_payload_rows(batch: &RecordBatch, key_column: usize, list_start: usize, keys: &[i32]) {
    assert_eq!(int32_array(batch, key_column).values().as_ref(), keys);
    let ints = batch.column(list_start).as_any().downcast_ref::<ListArray>().unwrap();
    let flags = batch.column(list_start + 1).as_any().downcast_ref::<ListArray>().unwrap();
    let tags = batch.column(list_start + 2).as_any().downcast_ref::<ListArray>().unwrap();
    let scores = batch.column(list_start + 3).as_any().downcast_ref::<ListArray>().unwrap();
    let int_values = ints.values().as_primitive::<Int64Type>();
    let flag_values = flags.values().as_any().downcast_ref::<BooleanArray>().unwrap();
    let tag_values = tags.values().as_any().downcast_ref::<StringArray>().unwrap();
    let score_values = scores.values().as_primitive::<Float32Type>();
    assert_eq!(int_values.values().as_ref(), &keys.iter().map(|key| i64::from(*key) * 10).collect::<Vec<_>>());
    assert_eq!(flag_values.iter().collect::<Vec<_>>(), keys.iter().map(|key| Some(key % 2 == 0)).collect::<Vec<_>>());
    assert_eq!(
        tag_values.iter().collect::<Vec<_>>(),
        keys.iter()
            .map(|key| Some(format!("tag-{key}")))
            .collect::<Vec<_>>()
            .iter()
            .map(|value| value.as_deref())
            .collect::<Vec<_>>()
    );
    assert_eq!(score_values.values().as_ref(), &keys.iter().map(|key| *key as f32 + 0.25).collect::<Vec<_>>());
}

pub(super) fn fixed_size_list_payload_batch(keys: &[i32]) -> RecordBatch {
    let child = Arc::new(Field::new("element", DataType::Float64, false));
    let values = keys.iter().flat_map(|key| [f64::from(*key) + 0.25, f64::from(*key) + 0.5]).collect::<Vec<_>>();
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("coordinates", DataType::FixedSizeList(Arc::clone(&child), 2), false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(keys.to_vec())),
            Arc::new(FixedSizeListArray::new(child, 2, Arc::new(Float64Array::from(values)), None)),
        ],
    )
    .unwrap()
}

pub(super) fn assert_fixed_size_list_payload_rows(
    batch: &RecordBatch,
    key_column: usize,
    list_column: usize,
    keys: &[i32],
) {
    assert_eq!(int32_array(batch, key_column).values().as_ref(), keys);
    let coordinates = batch.column(list_column).as_any().downcast_ref::<FixedSizeListArray>().unwrap();
    assert_eq!(coordinates.value_length(), 2);
    let values = coordinates.values().as_primitive::<Float64Type>();
    assert_eq!(
        values.values().as_ref(),
        &keys.iter().flat_map(|key| [f64::from(*key) + 0.25, f64::from(*key) + 0.5]).collect::<Vec<_>>()
    );
}

fn narrow_decimal_record_batch() -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("amount32", DataType::Decimal32(8, 2), true),
        Field::new("amount64", DataType::Decimal64(17, 4), true),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 4])),
            Arc::new(
                Decimal32Array::from(vec![Some(12345), None, Some(-5), Some(999999)])
                    .with_precision_and_scale(8, 2)
                    .unwrap(),
            ),
            Arc::new(
                Decimal64Array::from(vec![Some(-987654321), Some(42), None, Some(123456789)])
                    .with_precision_and_scale(17, 4)
                    .unwrap(),
            ),
        ],
    )
    .unwrap()
}

fn native_literal_batch_carries_decimal32_and_decimal64_with_exact_schemas() {
    let context = backend_context();
    let batch = narrow_decimal_record_batch();
    let schema = batch.schema();
    let mut builder = QueryPlanBuilder::new();
    let source =
        builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(Arc::clone(&schema), vec![batch.clone()])));
    let plan = builder.finish_with_host_sink(source).unwrap();

    let result = execute_query_plan(&context, &plan).unwrap();
    let QueryOutput::HostArrow(actual) = result.output() else {
        panic!("host sink should materialize narrow decimal literal output");
    };
    assert_record_batch_columns_equal(actual, &batch);
}

fn native_local_parquet_filters_projects_and_limits_decimal32_and_decimal64_payloads() {
    let dir = unique_temp_dir("query-engine-native-narrow-decimal");
    let path = dir.join("input.parquet");
    let context = backend_context();
    let batch = narrow_decimal_record_batch();
    let schema = batch.schema();
    let file = File::create(&path).unwrap();
    let mut writer =
        ArrowWriter::try_new(file, Arc::clone(&schema), Some(WriterProperties::builder().build())).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();

    let source = NativeLocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&schema),
        vec!["id".to_string(), "amount32".to_string(), "amount64".to_string()],
    );
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("small", DataType::Decimal32(8, 2), true),
        Field::new("large", DataType::Decimal64(17, 4), true),
    ]));
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let filter = builder
        .add_filter(source, Expr::binary(Expr::column("id"), BinaryOp::GtEq, Expr::literal(LiteralValue::Int32(2))))
        .unwrap();
    let projection = builder
        .add_projection(
            filter,
            projection_outputs(
                vec![Expr::alias(Expr::column("amount32"), "small"), Expr::alias(Expr::column("amount64"), "large")],
                Arc::clone(&output_schema),
            ),
        )
        .unwrap();
    let limit = builder.add_limit(projection, LimitSpec::fetch(2)).unwrap();
    let plan = builder.finish_with_host_sink_schema(limit, Arc::clone(&output_schema)).unwrap();

    let result = execute_query_plan(&context, &plan).unwrap();
    let QueryOutput::HostArrow(actual) = result.output() else {
        panic!("host sink should materialize narrow decimal Parquet output");
    };
    assert_eq!(actual.schema(), output_schema);
    assert_eq!(actual.column(0).as_primitive::<Decimal32Type>().iter().collect::<Vec<_>>(), vec![None, Some(-5)]);
    assert_eq!(actual.column(1).as_primitive::<Decimal64Type>().iter().collect::<Vec<_>>(), vec![Some(42), None]);

    fs::remove_dir_all(dir).unwrap();
}

fn native_parquet_narrow_decimal_storage_preserves_declared_narrow_or_decimal128_logical_width() {
    let cases = [
        (
            "decimal32",
            DataType::Decimal32(8, 2),
            Arc::new(Decimal32Array::from(vec![Some(12345), None, Some(-5)]).with_precision_and_scale(8, 2).unwrap())
                as ArrayRef,
            Arc::new(Decimal128Array::from(vec![Some(12345), None, Some(-5)]).with_precision_and_scale(8, 2).unwrap())
                as ArrayRef,
        ),
        (
            "decimal64",
            DataType::Decimal64(17, 4),
            Arc::new(
                Decimal64Array::from(vec![Some(-987654321), None, Some(42)]).with_precision_and_scale(17, 4).unwrap(),
            ) as ArrayRef,
            Arc::new(
                Decimal128Array::from(vec![Some(-987654321), None, Some(42)]).with_precision_and_scale(17, 4).unwrap(),
            ) as ArrayRef,
        ),
    ];

    for (name, narrow_type, narrow_array, wide_array) in cases {
        let dir = unique_temp_dir(&format!("query-engine-native-physical-logical-{name}"));
        let path = dir.join("input.parquet");
        let context = backend_context();
        let narrow_schema = Arc::new(Schema::new(vec![Field::new("amount", narrow_type.clone(), true)]));
        let narrow_batch = RecordBatch::try_new(Arc::clone(&narrow_schema), vec![narrow_array]).unwrap();
        let file = File::create(&path).unwrap();
        let mut writer =
            ArrowWriter::try_new(file, Arc::clone(&narrow_schema), Some(WriterProperties::builder().build())).unwrap();
        writer.write(&narrow_batch).unwrap();
        writer.close().unwrap();

        let execute = |schema: Arc<Schema>| {
            let source = NativeLocalParquetSource::from_files(
                vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
                Arc::clone(&schema),
                vec!["amount".to_string()],
            );
            let mut builder = QueryPlanBuilder::new();
            let source = builder.add_source(SourceNode::LocalParquet(source));
            let plan = builder.finish_with_host_sink_schema(source, schema).unwrap();
            let result = execute_query_plan(&context, &plan).unwrap();
            let QueryOutput::HostArrow(batch) = result.output() else {
                panic!("host sink should materialize decimal output");
            };
            batch.clone()
        };

        assert_record_batch_columns_equal(&execute(Arc::clone(&narrow_schema)), &narrow_batch);

        let (precision, scale) = match narrow_type {
            DataType::Decimal32(precision, scale) | DataType::Decimal64(precision, scale) => (precision, scale),
            _ => unreachable!(),
        };
        let wide_schema =
            Arc::new(Schema::new(vec![Field::new("amount", DataType::Decimal128(precision, scale), true)]));
        let wide_batch = RecordBatch::try_new(Arc::clone(&wide_schema), vec![wide_array]).unwrap();
        assert_record_batch_columns_equal(&execute(wide_schema), &wide_batch);

        fs::remove_dir_all(dir).unwrap();
    }
}

fn native_local_parquet_source_carries_timestamp_timezone_to_host_output() {
    let dir = unique_temp_dir("query-engine-native-timestamp-tz");
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
    let plan = builder.finish_with_host_sink(source).unwrap();

    let result = execute_query_plan(&context, &plan).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize timestamp-with-timezone output");
    };
    assert_eq!(batch.schema(), schema);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[1, 2, 3]);
    let event_ts = batch.column(1).as_primitive::<TimestampMicrosecondType>();
    assert_eq!(event_ts.values().as_ref(), &[1_700_000_000_000_000, 0, 1_700_000_000_000_123]);
    assert!(event_ts.is_valid(0));
    assert!(!event_ts.is_valid(1));
    assert!(event_ts.is_valid(2));

    fs::remove_dir_all(dir).unwrap();
}

fn native_literal_batch_carries_exact_basic_list_matrix() {
    let context = backend_context();
    let batch = basic_list_record_batch();
    let schema = batch.schema();
    let mut builder = QueryPlanBuilder::new();
    let source =
        builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(Arc::clone(&schema), vec![batch.clone()])));
    let plan = builder.finish_with_host_sink(source).unwrap();

    let result = execute_query_plan(&context, &plan).unwrap();
    let QueryOutput::HostArrow(actual) = result.output() else {
        panic!("host sink should materialize literal list output");
    };
    assert_record_batch_columns_equal(actual, &batch);
}

fn native_local_parquet_carries_exact_basic_list_matrix() {
    let dir = unique_temp_dir("query-engine-native-basic-list-matrix");
    let path = dir.join("input.parquet");
    let context = backend_context();
    let batch = basic_list_record_batch();
    let schema = batch.schema();
    let file = File::create(&path).unwrap();
    let mut writer =
        ArrowWriter::try_new(file, Arc::clone(&schema), Some(WriterProperties::builder().build())).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();

    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&schema),
        schema.fields().iter().map(|field| field.name().to_string()).collect(),
    );
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let plan = builder.finish_with_host_sink(source).unwrap();

    let result = execute_query_plan(&context, &plan).unwrap();
    let QueryOutput::HostArrow(actual) = result.output() else {
        panic!("host sink should materialize local Parquet list output");
    };
    assert_record_batch_columns_equal(actual, &batch);

    fs::remove_dir_all(dir).unwrap();
}

fn native_literal_batch_carries_sliced_float64_fixed_size_list_with_empty_input_batch() {
    let context = backend_context();
    let child = Arc::new(Field::new("element", DataType::Float64, false));
    let full = FixedSizeListArray::new(
        Arc::clone(&child),
        3,
        Arc::new(Float64Array::from(vec![
            99.0,
            98.0,
            97.0,
            f64::NAN,
            f64::INFINITY,
            -0.0,
            f64::NEG_INFINITY,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            90.0,
            91.0,
            92.0,
        ])),
        None,
    );
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("features", DataType::FixedSizeList(Arc::clone(&child), 3), false),
    ]));
    let expected = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(vec![1, 2, 3])), Arc::new(full.slice(1, 3))],
    )
    .unwrap();
    let empty = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(Vec::<i32>::new())),
            Arc::new(FixedSizeListArray::new(child, 3, Arc::new(Float64Array::from(Vec::<f64>::new())), None)),
        ],
    )
    .unwrap();
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(
        Arc::clone(&schema),
        vec![empty, expected.clone()],
    )));
    let plan = builder.finish_with_host_sink(source).unwrap();

    let result = execute_query_plan(&context, &plan).unwrap();
    let QueryOutput::HostArrow(actual) = result.output() else {
        panic!("host sink should materialize FixedSizeList literal output");
    };
    assert_record_batch_columns_equal(actual, &expected);
}

fn native_local_parquet_filters_projects_and_limits_nullable_int32_fixed_size_list() {
    let dir = unique_temp_dir("query-engine-native-fixed-size-list");
    let path = dir.join("input.parquet");
    let context = backend_context();
    let child = Arc::new(Field::new("element", DataType::Int32, true));
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("features", DataType::FixedSizeList(Arc::clone(&child), 2), true),
    ]));
    let features = FixedSizeListArray::new(
        child,
        2,
        Arc::new(Int32Array::from(vec![Some(1), None, Some(3), Some(4), Some(5), Some(6), Some(7), Some(8)])),
        Some(NullBuffer::from(vec![true, false, true, true])),
    );
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(vec![1, 2, 3, 4])), Arc::new(features)],
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
        vec!["id".to_string(), "features".to_string()],
    );
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("features", DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Int32, true)), 2), true),
    ]));
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let filter = builder
        .add_filter(source, Expr::binary(Expr::column("id"), BinaryOp::GtEq, Expr::literal(LiteralValue::Int32(2))))
        .unwrap();
    let projection = builder
        .add_projection(
            filter,
            projection_outputs(vec![Expr::column("id"), Expr::column("features")], Arc::clone(&output_schema)),
        )
        .unwrap();
    let limit = builder.add_limit(projection, LimitSpec::fetch(2)).unwrap();
    let plan = builder.finish_with_host_sink_schema(limit, Arc::clone(&output_schema)).unwrap();

    let result = execute_query_plan(&context, &plan).unwrap();
    let QueryOutput::HostArrow(actual) = result.output() else {
        panic!("host sink should materialize FixedSizeList Parquet output");
    };
    assert_eq!(actual.schema(), output_schema);
    assert_eq!(int32_array(actual, 0).values().as_ref(), &[2, 3]);
    let features = actual.column(1).as_any().downcast_ref::<FixedSizeListArray>().unwrap();
    assert_eq!(features.value_length(), 2);
    assert!(features.is_null(0));
    assert!(features.is_valid(1));
    assert_eq!(features.value(1).as_primitive::<Int32Type>().values(), &[5, 6]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn type_carry_cases_2() {
    run_named_cases(&[
        (
            "core_execution::type_carry::native_local_parquet_filters_projects_and_limits_nullable_int32_fixed_size_list",
            native_local_parquet_filters_projects_and_limits_nullable_int32_fixed_size_list,
        ),
        (
            "core_execution::type_carry::native_local_parquet_source_carries_large_list_large_utf8_to_host_output",
            native_local_parquet_source_carries_large_list_large_utf8_to_host_output,
        ),
        (
            "core_execution::type_carry::native_scalar_filter_projection_alias_sort_and_topk_carry_basic_list_payloads",
            native_scalar_filter_projection_alias_sort_and_topk_carry_basic_list_payloads,
        ),
        (
            "core_execution::type_carry::native_literal_batch_carries_sliced_float64_fixed_size_list_with_empty_input_batch",
            native_literal_batch_carries_sliced_float64_fixed_size_list_with_empty_input_batch,
        ),
        (
            "core_execution::type_carry::native_literal_batch_carries_exact_basic_list_matrix",
            native_literal_batch_carries_exact_basic_list_matrix,
        ),
        (
            "core_execution::type_carry::native_literal_batch_carries_decimal32_and_decimal64_with_exact_schemas",
            native_literal_batch_carries_decimal32_and_decimal64_with_exact_schemas,
        ),
    ]);
}

fn native_local_parquet_source_carries_list_utf8_through_scalar_filter_to_host_output() {
    let dir = unique_temp_dir("query-engine-native-list-utf8");
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
    let filter = builder.add_filter(source, filter).unwrap();
    let projection = builder.add_projection(filter, projection).unwrap();
    let plan = builder.finish_with_host_sink_schema(projection, Arc::clone(&output_schema)).unwrap();

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
fn type_carry_cases_1() {
    run_named_cases(&[
        (
            "core_execution::type_carry::native_local_parquet_source_carries_list_utf8_through_scalar_filter_to_host_output",
            native_local_parquet_source_carries_list_utf8_through_scalar_filter_to_host_output,
        ),
        (
            "core_execution::type_carry::native_parquet_narrow_decimal_storage_preserves_declared_narrow_or_decimal128_logical_width",
            native_parquet_narrow_decimal_storage_preserves_declared_narrow_or_decimal128_logical_width,
        ),
        (
            "core_execution::type_carry::native_local_parquet_source_carries_timestamp_timezone_to_host_output",
            native_local_parquet_source_carries_timestamp_timezone_to_host_output,
        ),
        (
            "core_execution::type_carry::native_local_parquet_filters_projects_and_limits_decimal32_and_decimal64_payloads",
            native_local_parquet_filters_projects_and_limits_decimal32_and_decimal64_payloads,
        ),
        (
            "core_execution::type_carry::native_local_parquet_carries_exact_basic_list_matrix",
            native_local_parquet_carries_exact_basic_list_matrix,
        ),
        (
            "core_execution::type_carry::native_local_parquet_limit_materializes_basic_list_shape",
            native_local_parquet_limit_materializes_basic_list_shape,
        ),
    ]);
}

fn native_scalar_filter_projection_alias_sort_and_topk_carry_basic_list_payloads() {
    let context = backend_context();
    let input = basic_list_payload_batch(&[3, 1, 4, 2]);
    let input_schema = input.schema();
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("sort_key", DataType::Int32, false),
        Field::new("integer_payload", input_schema.field(2).data_type().clone(), true),
        Field::new("boolean_payload", input_schema.field(3).data_type().clone(), true),
        Field::new("string_payload", input_schema.field(4).data_type().clone(), true),
        Field::new("float_payload", input_schema.field(5).data_type().clone(), true),
    ]));

    let execute = |limit: Option<usize>| {
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(
            Arc::clone(&input_schema),
            vec![input.clone()],
        )));
        let filter = builder
            .add_filter(
                source,
                Expr::binary(Expr::column("key"), BinaryOp::GtEq, Expr::literal(LiteralValue::Int32(2))),
            )
            .unwrap();
        let projection = builder
            .add_projection(
                filter,
                projection_outputs(
                    vec![
                        Expr::column("key"),
                        Expr::column("ints"),
                        Expr::column("flags"),
                        Expr::column("tags"),
                        Expr::column("scores"),
                    ],
                    Arc::clone(&output_schema),
                ),
            )
            .unwrap();
        let sort = builder
            .add_sort(projection, SortSpec::new(vec![SortExpr::new(Expr::column("sort_key"), false, false)]))
            .unwrap();
        let root = limit.map_or(sort, |rows| builder.add_limit(sort, LimitSpec::fetch(rows)).unwrap());
        let plan = builder.finish_with_host_sink(root).unwrap();
        let result = execute_query_plan(&context, &plan).unwrap();
        let QueryOutput::HostArrow(batch) = result.output() else {
            panic!("host sink should materialize carried list payloads");
        };
        batch.clone()
    };

    let sorted = execute(None);
    assert_eq!(sorted.schema(), output_schema);
    assert_basic_list_payload_rows(&sorted, 0, 1, &[2, 3, 4]);
    let topk = execute(Some(2));
    assert_eq!(topk.schema(), output_schema);
    assert_basic_list_payload_rows(&topk, 0, 1, &[2, 3]);
}

fn native_local_parquet_limit_materializes_basic_list_shape() {
    let dir = unique_temp_dir("query-engine-native-list-utf8-null-limit");
    let path = dir.join("input.parquet");
    let context = backend_context();
    let item = Arc::new(Field::new("element", DataType::Utf8, true));
    let schema = Arc::new(Schema::new(vec![
        Field::new("vertexId", DataType::Int32, false),
        Field::new("vertexAttributes__tag", DataType::List(Arc::clone(&item)), true),
        Field::new("vertexAttributes__label", DataType::List(Arc::clone(&item)), true),
        Field::new(
            "vertexAttributes__rank",
            DataType::List(Arc::new(Field::new("element", DataType::Int64, true))),
            true,
        ),
    ]));
    let null_strings = || {
        Arc::new(ListArray::new(
            Arc::clone(&item),
            OffsetBuffer::new(ScalarBuffer::from(vec![0; 13])),
            Arc::new(StringArray::from(Vec::<Option<&str>>::new())),
            Some(NullBuffer::from(vec![false; 12])),
        )) as ArrayRef
    };
    let ports = ListArray::new(
        Arc::new(Field::new("element", DataType::Int64, true)),
        OffsetBuffer::new(ScalarBuffer::from((0..=12).collect::<Vec<i32>>())),
        Arc::new(Int64Array::from_iter_values(100..112)),
        None,
    );
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from_iter_values(0..12)), null_strings(), null_strings(), Arc::new(ports)],
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
        schema.fields().iter().map(|field| field.name().to_string()).collect(),
    );
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let limit = builder.add_limit(source, LimitSpec::fetch(10)).unwrap();
    let plan = builder.finish_with_host_sink(limit).unwrap();

    let result = execute_query_plan(&context, &plan).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize all-null list output");
    };
    assert_eq!(batch.schema(), schema);
    assert_eq!(batch.num_rows(), 10);
    for index in [1, 2] {
        let lists = batch.column(index).as_any().downcast_ref::<ListArray>().unwrap();
        assert_eq!(lists.values().len(), 0);
        assert_eq!(lists.null_count(), 10);
    }
    let ports = batch.column(3).as_any().downcast_ref::<ListArray>().unwrap();
    assert_eq!(ports.len(), 10);
    assert_eq!(ports.values().as_primitive::<Int64Type>().values().as_ref(), &(100..110).collect::<Vec<_>>());

    fs::remove_dir_all(dir).unwrap();
}

fn native_local_parquet_source_carries_large_list_large_utf8_to_host_output() {
    let dir = unique_temp_dir("query-engine-native-large-list-utf8");
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
    let plan = builder.finish_with_host_sink(source).unwrap();

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
