use std::{fs::File, sync::Arc};

use arrow::array::{TimestampMillisecondArray, TimestampSecondArray};
use query_planning::{
    error::QueryError,
    expr::{CaseWhen, DatePart, UnaryMathOp},
};

use super::*;

fn native_query_projects_date64_and_time_types() {
    let context = backend_context();
    let input_schema = Arc::new(Schema::new(vec![
        Field::new("event_date64", DataType::Date64, false),
        Field::new("event_time_ms", DataType::Time32(TimeUnit::Millisecond), false),
        Field::new("event_time_ns", DataType::Time64(TimeUnit::Nanosecond), false),
    ]));
    let input_batch = RecordBatch::try_new(
        Arc::clone(&input_schema),
        vec![
            Arc::new(Date64Array::from(vec![1_577_836_800_000, 1_580_428_800_000])),
            Arc::new(Time32MillisecondArray::from(vec![1_000, 86_399_000])),
            Arc::new(Time64NanosecondArray::from(vec![1_000_000_000, 86_399_000_000_000])),
        ],
    )
    .unwrap();
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("event_date64", DataType::Date64, false),
        Field::new("event_time_ms", DataType::Time32(TimeUnit::Millisecond), false),
        Field::new("event_time_ns", DataType::Time64(TimeUnit::Nanosecond), false),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let source =
        builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", Arc::clone(&input_schema))));
    let projection = builder
        .add_projection(
            source,
            projection_outputs(
                vec![Expr::column("event_date64"), Expr::column("event_time_ms"), Expr::column("event_time_ns")],
                output_schema,
            ),
        )
        .unwrap();
    let plan = builder.finish_with_host_sink(projection).unwrap();
    let admitted = context.admit_plan(&plan).unwrap();
    let runtime = Arc::new(admitted.runtime().unwrap());

    let mut gpu_inputs = GpuResidentInputMap::new();
    gpu_inputs.insert(
        "input".to_string(),
        GpuResultHandle::new(
            &admitted,
            GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &input_batch).unwrap()),
            Arc::clone(&input_schema),
        )
        .unwrap(),
    );

    let result = execute_query_plan_with_gpu_inputs(&admitted, &plan, &gpu_inputs).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.schema().field(0).data_type(), &DataType::Date64);
    assert_eq!(batch.schema().field(1).data_type(), &DataType::Time32(TimeUnit::Millisecond));
    assert_eq!(batch.schema().field(2).data_type(), &DataType::Time64(TimeUnit::Nanosecond));
    assert_eq!(
        batch.column(0).as_any().downcast_ref::<Date64Array>().unwrap().values().as_ref(),
        &[1_577_836_800_000, 1_580_428_800_000]
    );
    assert_eq!(
        batch.column(1).as_any().downcast_ref::<Time32MillisecondArray>().unwrap().values().as_ref(),
        &[1_000, 86_399_000]
    );
    assert_eq!(
        batch.column(2).as_any().downcast_ref::<Time64NanosecondArray>().unwrap().values().as_ref(),
        &[1_000_000_000, 86_399_000_000_000]
    );
    admitted.finish();
}

fn project_gpu_resident_batch(
    input_batch: &RecordBatch,
    expressions: Vec<Expr>,
    output_schema: Arc<Schema>,
) -> RecordBatch {
    let context = backend_context();
    let input_schema = input_batch.schema();
    let mut builder = QueryPlanBuilder::new();
    let source =
        builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", Arc::clone(&input_schema))));
    let projection = builder.add_projection(source, projection_outputs(expressions, output_schema)).unwrap();
    let plan = builder.finish_with_host_sink(projection).unwrap();
    let admitted = context.admit_plan(&plan).unwrap();
    let runtime = Arc::new(admitted.runtime().unwrap());

    let mut gpu_inputs = GpuResidentInputMap::new();
    gpu_inputs.insert(
        "input".to_string(),
        GpuResultHandle::new(
            &admitted,
            GpuDataFrame::new(DataFrame::from_record_batch(&runtime, input_batch).unwrap()),
            input_schema,
        )
        .unwrap(),
    );

    let result = execute_query_plan_with_gpu_inputs(&admitted, &plan, &gpu_inputs).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    let batch = batch.clone();
    admitted.finish();
    batch
}

fn native_query_extracts_time_fields_from_every_timestamp_unit() {
    let units = ["s", "ms", "us", "ns"];
    let input_schema = Arc::new(Schema::new(vec![
        Field::new("ts_s", DataType::Timestamp(TimeUnit::Second, None), true),
        Field::new("ts_ms", DataType::Timestamp(TimeUnit::Millisecond, None), true),
        Field::new("ts_us", DataType::Timestamp(TimeUnit::Microsecond, None), true),
        Field::new("ts_ns", DataType::Timestamp(TimeUnit::Nanosecond, None), true),
    ]));
    // Rows: 2020-03-04 12:34:56.789123456, the tick before the epoch, NULL, and 00:00:59.
    let input_batch = RecordBatch::try_new(
        Arc::clone(&input_schema),
        vec![
            Arc::new(TimestampSecondArray::from(vec![Some(1_583_325_296), Some(-1), None, Some(59)])),
            Arc::new(TimestampMillisecondArray::from(vec![Some(1_583_325_296_789), Some(-1), None, Some(59_000)])),
            Arc::new(TimestampMicrosecondArray::from(vec![
                Some(1_583_325_296_789_123),
                Some(-1),
                None,
                Some(59_000_000),
            ])),
            Arc::new(TimestampNanosecondArray::from(vec![
                Some(1_583_325_296_789_123_456),
                Some(-1),
                None,
                Some(59_000_000_000),
            ])),
        ],
    )
    .unwrap();
    let fields = [
        ("hour", DatePart::Hour),
        ("minute", DatePart::Minute),
        ("second", DatePart::Second),
        ("millisecond", DatePart::Millisecond),
        ("microsecond", DatePart::Microsecond),
    ];
    let mut expressions = Vec::new();
    let mut output_fields = Vec::new();
    for unit in units {
        for (name, field) in fields {
            expressions.push(Expr::extract(field, Expr::column(format!("ts_{unit}"))));
            output_fields.push(Field::new(format!("{name}_{unit}"), DataType::Int32, true));
        }
    }

    let batch = project_gpu_resident_batch(&input_batch, expressions, Arc::new(Schema::new(output_fields)));

    // Per unit and field, the four rows in order. MILLISECOND and MICROSECOND
    // include the whole seconds of the minute, and finer remainders are dropped.
    let expected: [[[Option<i32>; 4]; 5]; 4] = [
        [
            [Some(12), Some(23), None, Some(0)],
            [Some(34), Some(59), None, Some(0)],
            [Some(56), Some(59), None, Some(59)],
            [Some(56_000), Some(59_000), None, Some(59_000)],
            [Some(56_000_000), Some(59_000_000), None, Some(59_000_000)],
        ],
        [
            [Some(12), Some(23), None, Some(0)],
            [Some(34), Some(59), None, Some(0)],
            [Some(56), Some(59), None, Some(59)],
            [Some(56_789), Some(59_999), None, Some(59_000)],
            [Some(56_789_000), Some(59_999_000), None, Some(59_000_000)],
        ],
        [
            [Some(12), Some(23), None, Some(0)],
            [Some(34), Some(59), None, Some(0)],
            [Some(56), Some(59), None, Some(59)],
            [Some(56_789), Some(59_999), None, Some(59_000)],
            [Some(56_789_123), Some(59_999_999), None, Some(59_000_000)],
        ],
        [
            [Some(12), Some(23), None, Some(0)],
            [Some(34), Some(59), None, Some(0)],
            [Some(56), Some(59), None, Some(59)],
            [Some(56_789), Some(59_999), None, Some(59_000)],
            [Some(56_789_123), Some(59_999_999), None, Some(59_000_000)],
        ],
    ];
    for (unit_index, unit) in units.iter().enumerate() {
        for (field_index, (name, _)) in fields.iter().enumerate() {
            let column = unit_index * fields.len() + field_index;
            assert_eq!(batch.schema().field(column).data_type(), &DataType::Int32, "{name} over {unit}");
            assert_eq!(batch.column(column).null_count(), 1, "{name} over {unit}");
            assert_eq!(
                int32_array(&batch, column).iter().collect::<Vec<_>>(),
                expected[unit_index][field_index],
                "{name} over {unit}"
            );
        }
    }
}

fn native_query_extracts_minute_from_integer_second_cast_including_negative_seconds() {
    let input_schema = Arc::new(Schema::new(vec![Field::new("epoch_s", DataType::Int64, true)]));
    let input_batch = RecordBatch::try_new(
        Arc::clone(&input_schema),
        vec![Arc::new(Int64Array::from(vec![
            Some(-1),
            Some(-61),
            Some(-3_600),
            Some(0),
            Some(59),
            Some(60),
            Some(3_599),
            None,
        ]))],
    )
    .unwrap();
    let output_schema = Arc::new(Schema::new(vec![Field::new("minute", DataType::Int32, true)]));

    let batch = project_gpu_resident_batch(
        &input_batch,
        vec![Expr::extract(
            DatePart::Minute,
            Expr::cast(Expr::column("epoch_s"), DataType::Timestamp(TimeUnit::Second, None)),
        )],
        output_schema,
    );

    assert_eq!(batch.schema().field(0).data_type(), &DataType::Int32);
    assert_eq!(
        int32_array(&batch, 0).iter().collect::<Vec<_>>(),
        [Some(59), Some(58), Some(0), Some(0), Some(0), Some(1), Some(59), None]
    );
}

fn native_query_counts_string_characters_and_bytes() {
    let context = backend_context();
    let input_schema = Arc::new(Schema::new(vec![
        Field::new("name_utf8", DataType::Utf8, true),
        Field::new("name_large", DataType::LargeUtf8, true),
        Field::new("name_view", DataType::Utf8View, true),
    ]));
    // Every non-empty value has more bytes than characters, so a unit mix-up changes each of them.
    let names = vec![Some("é"), Some("中"), Some("🙂"), Some("e\u{301}"), Some(""), None];
    let input_batch = RecordBatch::try_new(
        Arc::clone(&input_schema),
        vec![
            Arc::new(StringArray::from(names.clone())),
            Arc::new(LargeStringArray::from(names.clone())),
            Arc::new(StringViewArray::from(names)),
        ],
    )
    .unwrap();
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("chars_utf8", DataType::Int32, true),
        Field::new("bytes_utf8", DataType::Int32, true),
        Field::new("chars_large", DataType::Int64, true),
        Field::new("bytes_large", DataType::Int64, true),
        Field::new("chars_view", DataType::Int32, true),
        Field::new("bytes_view", DataType::Int32, true),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let source =
        builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", Arc::clone(&input_schema))));
    let projection = builder
        .add_projection(
            source,
            projection_outputs(
                vec![
                    Expr::string_length(Expr::column("name_utf8"), DataType::Int32),
                    Expr::octet_length(Expr::column("name_utf8"), DataType::Int32),
                    Expr::string_length(Expr::column("name_large"), DataType::Int64),
                    Expr::octet_length(Expr::column("name_large"), DataType::Int64),
                    Expr::string_length(Expr::column("name_view"), DataType::Int32),
                    Expr::octet_length(Expr::column("name_view"), DataType::Int32),
                ],
                output_schema,
            ),
        )
        .unwrap();
    let plan = builder.finish_with_host_sink(projection).unwrap();
    let admitted = context.admit_plan(&plan).unwrap();
    let runtime = Arc::new(admitted.runtime().unwrap());

    let mut gpu_inputs = GpuResidentInputMap::new();
    gpu_inputs.insert(
        "input".to_string(),
        GpuResultHandle::new(
            &admitted,
            GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &input_batch).unwrap()),
            Arc::clone(&input_schema),
        )
        .unwrap(),
    );

    let result = execute_query_plan_with_gpu_inputs(&admitted, &plan, &gpu_inputs).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    let expected_types =
        [DataType::Int32, DataType::Int32, DataType::Int64, DataType::Int64, DataType::Int32, DataType::Int32];
    for (index, expected) in expected_types.iter().enumerate() {
        assert_eq!(batch.schema().field(index).data_type(), expected, "column {index}");
        assert_eq!(batch.column(index).null_count(), 1, "column {index}");
    }
    let characters = [Some(1), Some(1), Some(1), Some(2), Some(0), None];
    let bytes = [Some(2), Some(3), Some(4), Some(3), Some(0), None];
    for index in [0, 4] {
        assert_eq!(int32_array(batch, index).iter().collect::<Vec<_>>(), characters, "characters column {index}");
    }
    for index in [1, 5] {
        assert_eq!(int32_array(batch, index).iter().collect::<Vec<_>>(), bytes, "bytes column {index}");
    }
    assert_eq!(int64_array(batch, 2).iter().collect::<Vec<_>>(), characters.map(|value| value.map(i64::from)));
    assert_eq!(int64_array(batch, 3).iter().collect::<Vec<_>>(), bytes.map(|value| value.map(i64::from)));
    admitted.finish();
}

fn native_query_executes_bitwise_scalar_expressions() {
    let dir = unique_temp_dir("query-engine-native-execution-bitwise-scalars");
    let path = dir.join("input.parquet");
    let context = backend_context();
    let runtime = Arc::new(context.runtime().unwrap());
    let input_schema = write_scalar_expression_parquet(&runtime, &path);
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("flags", DataType::Int64, false),
        Field::new("flags_and", DataType::Int64, false),
        Field::new("flags_or", DataType::Int64, false),
        Field::new("flags_xor", DataType::Int64, false),
        Field::new("flags_shift_left", DataType::Int64, false),
        Field::new("flags_shift_right", DataType::Int64, false),
        Field::new("flags_grouping", DataType::Int32, false),
    ]));
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["orderkey".to_string()],
    );
    let projection = vec![
        Expr::column("orderkey").alias("flags"),
        Expr::binary(Expr::column("orderkey"), BinaryOp::BitwiseAnd, Expr::literal(LiteralValue::Int64(3)))
            .alias("flags_and"),
        Expr::binary(Expr::column("orderkey"), BinaryOp::BitwiseOr, Expr::literal(LiteralValue::Int64(8)))
            .alias("flags_or"),
        Expr::binary(Expr::column("orderkey"), BinaryOp::BitwiseXor, Expr::literal(LiteralValue::Int64(1)))
            .alias("flags_xor"),
        Expr::binary(Expr::column("orderkey"), BinaryOp::ShiftLeft, Expr::literal(LiteralValue::Int64(1)))
            .alias("flags_shift_left"),
        Expr::binary(Expr::column("orderkey"), BinaryOp::ShiftRight, Expr::literal(LiteralValue::Int64(1)))
            .alias("flags_shift_right"),
        Expr::cast(
            Expr::binary(
                Expr::binary(Expr::column("orderkey"), BinaryOp::BitwiseAnd, Expr::literal(LiteralValue::Int64(2))),
                BinaryOp::ShiftRight,
                Expr::literal(LiteralValue::Int64(1)),
            ),
            DataType::Int32,
        )
        .alias("flags_grouping"),
    ];

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let projection = builder.add_projection(source, projection_outputs(projection, output_schema)).unwrap();
    let plan = builder.finish_with_host_sink(projection).unwrap();

    let result = execute_query_plan(&context, &plan).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(int64_array(batch, 0).values().as_ref(), &[1, 2, 3, 4]);
    assert_eq!(int64_array(batch, 1).values().as_ref(), &[1, 2, 3, 0]);
    assert_eq!(int64_array(batch, 2).values().as_ref(), &[9, 10, 11, 12]);
    assert_eq!(int64_array(batch, 3).values().as_ref(), &[0, 3, 2, 5]);
    assert_eq!(int64_array(batch, 4).values().as_ref(), &[2, 4, 6, 8]);
    assert_eq!(int64_array(batch, 5).values().as_ref(), &[0, 1, 1, 2]);
    assert_eq!(int32_array(batch, 6).values().as_ref(), &[0, 1, 1, 0]);

    fs::remove_dir_all(dir).unwrap();
}

fn native_query_executes_regexp_replace_projection_with_backrefs() {
    let dir = unique_temp_dir("query-engine-native-execution-regexp-replace");
    let path = dir.join("referers.parquet");
    let context = backend_context();
    let input_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("referer", DataType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&input_schema),
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 3, 4])),
            Arc::new(StringArray::from(vec![
                Some("https://www.example.com/path"),
                Some("http://rapids.ai/docs"),
                Some("mailto:plain"),
                None,
            ])),
        ],
    )
    .unwrap();
    let file = File::create(&path).unwrap();
    let mut writer = ArrowWriter::try_new(file, Arc::clone(&input_schema), None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();

    let output_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("domain", DataType::Utf8, true),
    ]));
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["id".to_string(), "referer".to_string()],
    );
    let projection = vec![
        Expr::column("id"),
        Expr::regexp_replace(Expr::column("referer"), r"^https?://(?:www\.)?([^/]+)/.*$", "\\1", DataType::Utf8)
            .alias("domain"),
    ];

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let projection = builder.add_projection(source, projection_outputs(projection, output_schema)).unwrap();
    let plan = builder.finish_with_host_sink(projection).unwrap();

    let result = execute_query_plan(&context, &plan).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(int64_array(batch, 0).values().as_ref(), &[1, 2, 3, 4]);
    assert_eq!(string_array(batch, 1).value(0), "example.com");
    assert_eq!(string_array(batch, 1).value(1), "rapids.ai");
    assert_eq!(string_array(batch, 1).value(2), "mailto:plain");
    assert!(string_array(batch, 1).is_null(3));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_query_rejects_scalar_type_mismatch_before_cudf_execution() {
    let path = Path::new("/missing/scalar-validation.parquet");
    let context = backend_context();
    let input_schema = Arc::new(Schema::new(vec![
        Field::new("orderkey", DataType::Int64, false),
        Field::new("shipdate", DataType::Date32, false),
        Field::new("event_ts", DataType::Timestamp(TimeUnit::Nanosecond, None), false),
        Field::new("status", DataType::Utf8, true),
        Field::new("amount", DataType::Decimal128(15, 2), false),
    ]));
    let output_schema = Arc::new(Schema::new(vec![Field::new("bad_like", DataType::Boolean, true)]));
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(path)],
        Arc::clone(&input_schema),
        vec!["orderkey".to_string()],
    );

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let projection = builder
        .add_projection(
            source,
            projection_outputs(vec![Expr::like(Expr::column("orderkey"), "1%").alias("bad_like")], output_schema),
        )
        .unwrap();
    let plan = builder.finish_with_host_sink(projection).unwrap();

    let error = match execute_query_plan(&context, &plan) {
        Ok(_) => panic!("native plan validation should reject LIKE over an integral input before cuDF execution"),
        Err(error) => error,
    };
    assert_eq!(error.code(), QueryErrorCode::UnsupportedType);
    assert_eq!(error.operation(), "query_engine.plan.validate_expr");
    assert_eq!(error.fact_value("context"), Some("LIKE input"));
}

fn native_query_truncates_nanosecond_timestamps_to_fixed_duration_fields() {
    let dir = unique_temp_dir("query-engine-native-execution-date-trunc-fixed-duration");
    let path = dir.join("tpch-like.parquet");
    let context = backend_context();
    let runtime = Arc::new(context.runtime().unwrap());
    let input_schema = write_scalar_expression_parquet(&runtime, &path);
    let nanoseconds = DataType::Timestamp(TimeUnit::Nanosecond, None);
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("to_nanosecond", nanoseconds.clone(), false),
        Field::new("to_microsecond", nanoseconds.clone(), false),
    ]));
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["event_ts".to_string()],
    );

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let projection = builder
        .add_projection(
            source,
            projection_outputs(
                vec![
                    Expr::date_trunc(DatePart::Nanosecond, Expr::column("event_ts"), nanoseconds.clone())
                        .alias("to_nanosecond"),
                    Expr::date_trunc(DatePart::Microsecond, Expr::column("event_ts"), nanoseconds)
                        .alias("to_microsecond"),
                ],
                output_schema,
            ),
        )
        .unwrap();
    let plan = builder.finish_with_host_sink(projection).unwrap();

    let result = execute_query_plan(&context, &plan).expect("cuDF floors timestamps to fixed-duration fields");
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    let values = |column: usize| {
        batch.column(column).as_any().downcast_ref::<TimestampNanosecondArray>().unwrap().values().to_vec()
    };
    assert_eq!(values(0), vec![100, 200, 300, 400]);
    assert_eq!(values(1), vec![0, 0, 0, 0]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_query_rejects_date_trunc_output_unit_mismatch_before_cudf_execution() {
    let path = Path::new("/missing/scalar-validation.parquet");
    let context = backend_context();
    let input_schema = Arc::new(Schema::new(vec![
        Field::new("orderkey", DataType::Int64, false),
        Field::new("shipdate", DataType::Date32, false),
        Field::new("event_ts", DataType::Timestamp(TimeUnit::Nanosecond, None), false),
        Field::new("status", DataType::Utf8, true),
        Field::new("amount", DataType::Decimal128(15, 2), false),
    ]));
    let output_schema =
        Arc::new(Schema::new(vec![Field::new("bad_trunc", DataType::Timestamp(TimeUnit::Millisecond, None), false)]));
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(path)],
        Arc::clone(&input_schema),
        vec!["event_ts".to_string()],
    );

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let projection = builder
        .add_projection(
            source,
            projection_outputs(
                vec![
                    Expr::date_trunc(
                        DatePart::Microsecond,
                        Expr::column("event_ts"),
                        DataType::Timestamp(TimeUnit::Millisecond, None),
                    )
                    .alias("bad_trunc"),
                ],
                output_schema,
            ),
        )
        .unwrap();
    let plan = builder.finish_with_host_sink(projection).unwrap();

    let error = match execute_query_plan(&context, &plan) {
        Ok(_) => panic!("native plan validation should reject DATE_TRUNC output unit mismatches"),
        Err(error) => error,
    };
    assert_eq!(error.code(), QueryErrorCode::UnsupportedType);
    assert_eq!(error.operation(), "query_engine.plan.validate_expr");
    assert_eq!(error.fact_value("context"), Some("DATE_TRUNC input/output"));
    assert_eq!(error.fact_value("input_type"), Some("Timestamp(Nanosecond, None)"));
    assert_eq!(error.fact_value("output_type"), Some("Timestamp(Millisecond, None)"));
}

#[test]
fn native_query_rejects_implicit_decimal_float_arithmetic_before_cudf_execution() {
    let path = Path::new("/missing/scalar-validation.parquet");
    let context = backend_context();
    let input_schema = Arc::new(Schema::new(vec![
        Field::new("orderkey", DataType::Int64, false),
        Field::new("shipdate", DataType::Date32, false),
        Field::new("event_ts", DataType::Timestamp(TimeUnit::Nanosecond, None), false),
        Field::new("status", DataType::Utf8, true),
        Field::new("amount", DataType::Decimal128(15, 2), false),
    ]));
    let output_schema = Arc::new(Schema::new(vec![Field::new("bad_amount", DataType::Decimal128(15, 2), true)]));
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(path)],
        Arc::clone(&input_schema),
        vec!["amount".to_string()],
    );

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let projection = builder
        .add_projection(
            source,
            projection_outputs(
                vec![
                    Expr::binary(Expr::column("amount"), BinaryOp::Add, Expr::literal(LiteralValue::Float64(1.0)))
                        .alias("bad_amount"),
                ],
                output_schema,
            ),
        )
        .unwrap();
    let plan = builder.finish_with_host_sink(projection).unwrap();

    let error = match execute_query_plan(&context, &plan) {
        Ok(_) => panic!("native plan validation should reject implicit Decimal128/Float64 arithmetic"),
        Err(error) => error,
    };
    assert_eq!(error.code(), QueryErrorCode::UnsupportedType);
    assert_eq!(error.operation(), "query_engine.plan.validate_expr");
    assert_eq!(error.fact_value("context"), Some("binary arithmetic operands"));
    assert_eq!(error.fact_value("left_type"), Some("Decimal128(15, 2)"));
    assert_eq!(error.fact_value("right_type"), Some("Float64"));
    assert!(error.fact_value("path").is_none());
}

#[test]
fn native_query_rejects_untyped_decimal_division_before_cudf_execution() {
    let dir = unique_temp_dir("query-engine-native-execution-untyped-decimal-divide-reject");
    let path = dir.join("input.parquet");
    let context = backend_context();
    let input_schema = write_decimal_ratio_parquet(&path);
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["numerator".to_string(), "denominator".to_string()],
    );
    let untyped_divide = Expr::binary(
        Expr::binary(Expr::column("numerator"), BinaryOp::Divide, Expr::column("denominator")),
        BinaryOp::Gt,
        Expr::literal(LiteralValue::Decimal128 { raw: 0, precision: 38, scale: 8 }),
    );

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let filter = builder.add_filter(source, untyped_divide).unwrap();
    let plan = builder.finish_with_host_sink(filter).unwrap();

    let error = match execute_query_plan(&context, &plan) {
        Ok(_) => panic!("native plan validation should reject decimal division without a declared output type"),
        Err(error) => error,
    };
    assert_eq!(error.code(), QueryErrorCode::UnsupportedType);
    assert_eq!(error.operation(), "query_engine.plan.validate_expr");
    assert_eq!(error.fact_value("left_type"), Some("Decimal128(38, 8)"));
    assert_eq!(error.fact_value("right_type"), Some("Decimal128(38, 8)"));
    assert_eq!(error.fact_value("operator"), Some("/"));

    fs::remove_dir_all(dir).unwrap();
}

fn native_query_executes_round_scalar_expression() {
    let dir = unique_temp_dir("query-engine-native-execution-round-scalar");
    let path = dir.join("input.parquet");
    let context = backend_context();
    let input_schema = write_decimal_ratio_parquet(&path);
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("rounded_literal", DataType::Float64, false),
        Field::new("rounded_ratio", DataType::Decimal128(38, 1), false),
    ]));
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["numerator".to_string(), "denominator".to_string()],
    );
    let ratio = Expr::binary_with_output_type(
        Expr::column("numerator"),
        BinaryOp::Divide,
        Expr::column("denominator"),
        DataType::Decimal128(38, 8),
    );

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let projection = builder
        .add_projection(
            source,
            projection_outputs(
                vec![
                    Expr::round(Expr::literal(LiteralValue::Float64(1.235)), 2, DataType::Float64)
                        .alias("rounded_literal"),
                    Expr::round(ratio, 1, DataType::Decimal128(38, 1)).alias("rounded_ratio"),
                ],
                output_schema,
            ),
        )
        .unwrap();
    let plan = builder.finish_with_host_sink(projection).unwrap();

    let result = execute_query_plan(&context, &plan).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 2);
    assert_eq!(float64_array(batch, 0).values().as_ref(), &[1.24, 1.24]);
    let rounded_ratio = decimal128_array(batch, 1);
    assert_eq!(rounded_ratio.data_type(), &DataType::Decimal128(38, 1));
    assert_eq!(rounded_ratio.values().as_ref(), &[3, 5]);

    fs::remove_dir_all(dir).unwrap();
}

fn native_query_executes_round_decimal_cast_to_integral_expression() {
    let dir = unique_temp_dir("query-engine-native-execution-round-decimal-integral-cast");
    let path = dir.join("input.parquet");
    let context = backend_context();
    let input_schema = write_decimal_ratio_parquet(&path);
    let output_schema = Arc::new(Schema::new(vec![Field::new("rounded_segment", DataType::Int32, false)]));
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["numerator".to_string(), "denominator".to_string()],
    );
    let ratio = Expr::binary_with_output_type(
        Expr::column("numerator"),
        BinaryOp::Divide,
        Expr::column("denominator"),
        DataType::Decimal128(38, 8),
    );
    let rounded_segment =
        Expr::cast(Expr::round(ratio, 0, DataType::Decimal128(38, 0)), DataType::Int32).alias("rounded_segment");

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let projection = builder.add_projection(source, projection_outputs(vec![rounded_segment], output_schema)).unwrap();
    let plan = builder.finish_with_host_sink(projection).unwrap();

    let result = execute_query_plan(&context, &plan).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 2);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[0, 1]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn scalar_expressions_cases_2() {
    run_named_cases(&[
        (
            "projection_validation::scalar_expressions::native_query_executes_round_decimal_cast_to_integral_expression",
            native_query_executes_round_decimal_cast_to_integral_expression,
        ),
        (
            "projection_validation::scalar_expressions::native_query_range_checked_casts_fail_the_query_outside_the_target_range",
            native_query_range_checked_casts_fail_the_query_outside_the_target_range,
        ),
        (
            "projection_validation::scalar_expressions::native_query_executes_regexp_replace_projection_with_backrefs",
            native_query_executes_regexp_replace_projection_with_backrefs,
        ),
        (
            "projection_validation::scalar_expressions::native_query_executes_bitwise_scalar_expressions",
            native_query_executes_bitwise_scalar_expressions,
        ),
        (
            "projection_validation::scalar_expressions::native_query_truncates_nanosecond_timestamps_to_fixed_duration_fields",
            native_query_truncates_nanosecond_timestamps_to_fixed_duration_fields,
        ),
        (
            "projection_validation::scalar_expressions::native_query_projects_date64_and_time_types",
            native_query_projects_date64_and_time_types,
        ),
        (
            "projection_validation::scalar_expressions::native_query_extracts_time_fields_from_every_timestamp_unit",
            native_query_extracts_time_fields_from_every_timestamp_unit,
        ),
        (
            "projection_validation::scalar_expressions::native_query_extracts_minute_from_integer_second_cast_including_negative_seconds",
            native_query_extracts_minute_from_integer_second_cast_including_negative_seconds,
        ),
        (
            "projection_validation::scalar_expressions::native_query_counts_string_characters_and_bytes",
            native_query_counts_string_characters_and_bytes,
        ),
    ]);
}

#[test]
fn native_query_rejects_date64_temporal_scalars_before_source_read() {
    let context = backend_context();
    let input_schema = Arc::new(Schema::new(vec![Field::new("event_date64", DataType::Date64, true)]));

    let cases = [
        (
            "extract_date64",
            vec![Expr::extract(DatePart::Year, Expr::column("event_date64")).alias("event_year")],
            Arc::new(Schema::new(vec![Field::new("event_year", DataType::Int32, true)])),
        ),
        (
            "interval_date64",
            vec![
                Expr::binary(
                    Expr::column("event_date64"),
                    BinaryOp::Add,
                    Expr::literal(LiteralValue::IntervalMonthDayNano { months: 0, days: 1, nanoseconds: 0 }),
                )
                .alias("event_plus_one_day"),
            ],
            Arc::new(Schema::new(vec![Field::new(
                "event_plus_one_day",
                DataType::Timestamp(TimeUnit::Millisecond, None),
                true,
            )])),
        ),
    ];

    for (name, projection, output_schema) in cases {
        let source = LocalParquetSource::from_path(
            format!("/tmp/does-not-need-to-exist-for-date64-{name}.parquet"),
            Arc::clone(&input_schema),
            vec!["event_date64".to_string()],
        );
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(source));
        let projection = builder.add_projection(source, projection_outputs(projection, output_schema)).unwrap();
        let plan = builder.finish_with_host_sink(projection).unwrap();

        let error = match execute_query_plan(&context, &plan) {
            Ok(_) => panic!("{name} should reject Date64 before source read"),
            Err(error) => error,
        };
        assert_eq!(error.code(), QueryErrorCode::UnsupportedType, "{name}");
        assert!(error.fact_value("path").is_none(), "{name} should fail before source read");
    }
}

#[test]
fn native_query_rejects_time_field_extraction_over_date32_and_timezone_before_source_read() {
    let context = backend_context();
    let cases = [
        ("date32", DataType::Date32),
        ("timestamp_with_timezone", DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into()))),
    ];

    for (name, input_type) in cases {
        let input_schema = Arc::new(Schema::new(vec![Field::new("event", input_type, true)]));
        let source = LocalParquetSource::from_path(
            format!("/tmp/does-not-need-to-exist-for-extract-time-{name}.parquet"),
            Arc::clone(&input_schema),
            vec!["event".to_string()],
        );
        let output_schema = Arc::new(Schema::new(vec![Field::new("event_hour", DataType::Int32, true)]));
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(source));
        let projection = builder
            .add_projection(
                source,
                projection_outputs(
                    vec![Expr::extract(DatePart::Hour, Expr::column("event")).alias("event_hour")],
                    output_schema,
                ),
            )
            .unwrap();
        let plan = builder.finish_with_host_sink(projection).unwrap();

        let error = match execute_query_plan(&context, &plan) {
            Ok(_) => panic!("{name} should reject time field extraction before source read"),
            Err(error) => error,
        };
        assert_eq!(error.code(), QueryErrorCode::UnsupportedType, "{name}");
        assert_eq!(error.fact_value("context"), Some("EXTRACT time field input"), "{name}");
        assert!(error.fact_value("path").is_none(), "{name} should fail before source read");
    }
}

#[test]
fn native_query_rejects_out_of_envelope_substring_before_source_read() {
    let context = backend_context();
    let input_schema = Arc::new(Schema::new(vec![
        Field::new("orderkey", DataType::Int64, false),
        Field::new("status", DataType::Utf8, true),
    ]));

    let cases = [
        (
            "zero_start",
            Expr::substring(Expr::column("status"), 0, Some(1)).alias("bad_substring"),
            QueryErrorCode::UnsupportedExpression,
            None,
        ),
        (
            "negative_length",
            Expr::substring(Expr::column("status"), 1, Some(-1)).alias("bad_substring"),
            QueryErrorCode::UnsupportedExpression,
            None,
        ),
        (
            "stop_overflow",
            Expr::substring(Expr::column("status"), i32::MAX, Some(2)).alias("bad_substring"),
            QueryErrorCode::UnsupportedExpression,
            None,
        ),
        (
            "non_string_source",
            Expr::substring(Expr::column("orderkey"), 1, Some(1)).alias("bad_substring"),
            QueryErrorCode::UnsupportedType,
            Some("SUBSTRING input"),
        ),
    ];

    for (name, expr, expected_code, expected_context) in cases {
        let source = LocalParquetSource::from_path(
            format!("/tmp/does-not-need-to-exist-for-substring-{name}.parquet"),
            Arc::clone(&input_schema),
            vec!["orderkey".to_string(), "status".to_string()],
        );
        let output_schema = Arc::new(Schema::new(vec![Field::new("bad_substring", DataType::Utf8, true)]));

        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(source));
        let projection = builder.add_projection(source, projection_outputs(vec![expr], output_schema)).unwrap();
        let plan = builder.finish_with_host_sink(projection).unwrap();

        let error = match execute_query_plan(&context, &plan) {
            Ok(_) => panic!("{name} should reject before source read"),
            Err(error) => error,
        };
        assert_eq!(error.code(), expected_code, "{name}");
        assert_eq!(error.operation(), "query_engine.plan.validate_expr", "{name}");
        assert_eq!(error.fact_value("context"), expected_context, "{name}");
        assert!(error.fact_value("path").is_none(), "{name} should fail before source read");
    }
}

#[test]
fn native_query_rejects_unread_source_column_in_nested_scalar_before_cudf_execution() {
    let path = Path::new("/missing/scalar-validation.parquet");
    let context = backend_context();
    let input_schema = Arc::new(Schema::new(vec![
        Field::new("orderkey", DataType::Int64, false),
        Field::new("shipdate", DataType::Date32, false),
        Field::new("event_ts", DataType::Timestamp(TimeUnit::Nanosecond, None), false),
        Field::new("status", DataType::Utf8, true),
        Field::new("amount", DataType::Decimal128(15, 2), false),
    ]));
    let output_schema = Arc::new(Schema::new(vec![Field::new("status_label", DataType::Utf8, true)]));
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(path)],
        Arc::clone(&input_schema),
        vec!["orderkey".to_string()],
    );
    let label = Expr::case(
        vec![CaseWhen::new(
            Expr::like(Expr::column("status"), "O%"),
            Expr::literal(LiteralValue::Utf8("open".to_string())),
        )],
        Expr::literal(LiteralValue::Utf8("other".to_string())),
    )
    .alias("status_label");

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let projection = builder.add_projection(source, projection_outputs(vec![label], output_schema)).unwrap();
    let plan = builder.finish_with_host_sink(projection).unwrap();

    let error = match execute_query_plan(&context, &plan) {
        Ok(_) => panic!("native plan validation should reject unread nested scalar columns before cuDF execution"),
        Err(error) => error,
    };
    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.operation(), "query_engine.plan.validate_expr");
    assert_eq!(error.fact_value("column"), Some("status"));
    assert!(error.fact_value("path").is_none());
}

fn project_over_parquet(
    name: &str,
    input: RecordBatch,
    exprs: Vec<Expr>,
    output_schema: Arc<Schema>,
) -> QueryEngineResult<RecordBatch> {
    let dir = unique_temp_dir(name);
    let path = dir.join("input.parquet");
    let input_schema = input.schema();
    let mut writer = ArrowWriter::try_new(File::create(&path).unwrap(), Arc::clone(&input_schema), None).unwrap();
    writer.write(&input).unwrap();
    writer.close().unwrap();
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        input_schema.fields().iter().map(|field| field.name().clone()).collect(),
    );
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let projection = builder.add_projection(source, projection_outputs(exprs, output_schema)).unwrap();
    let plan = builder.finish_with_host_sink(projection).unwrap();
    let result = execute_query_plan(&backend_context(), &plan);
    fs::remove_dir_all(dir).unwrap();
    let result = result?;
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    Ok(batch.clone())
}

fn assert_cudf_input_failure(error: &QueryError, cudf_code: &str) {
    assert_eq!(error.code(), QueryErrorCode::CudfInvalidInput, "got {error:?}");
    assert_eq!(error.fact_value("cudf_code"), Some(cudf_code), "got {error:?}");
}

fn native_query_casts_integers_between_widths_and_formats_them_as_text() {
    let input = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("wide", DataType::Int64, true),
            Field::new("rank", DataType::UInt64, true),
            Field::new("narrow", DataType::Int32, true),
            Field::new("low", DataType::Int64, true),
            Field::new("top", DataType::UInt64, true),
        ])),
        vec![
            Arc::new(Int64Array::from(vec![Some(i64::from(i32::MAX)), Some(i64::from(i32::MIN)), Some(0), None])),
            Arc::new(UInt64Array::from(vec![Some(0), Some(i64::MAX as u64), Some(7), None])),
            Arc::new(Int32Array::from(vec![Some(255), Some(0), Some(7), None])),
            Arc::new(Int64Array::from(vec![Some(i64::MIN), Some(-1), Some(0), None])),
            Arc::new(UInt64Array::from(vec![Some(u64::MAX), Some(0), Some(1), None])),
        ],
    )
    .unwrap();
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("wide_i32", DataType::Int32, true),
        Field::new("rank_i64", DataType::Int64, true),
        Field::new("narrow_u8", DataType::UInt8, true),
        Field::new("low_text", DataType::Utf8, true),
        Field::new("top_text", DataType::Utf8, true),
    ]));
    let exprs = vec![
        Expr::cast(Expr::column("wide"), DataType::Int32),
        Expr::cast(Expr::column("rank"), DataType::Int64),
        Expr::cast(Expr::column("narrow"), DataType::UInt8),
        Expr::cast(Expr::column("low"), DataType::Utf8),
        Expr::cast(Expr::column("top"), DataType::Utf8),
    ];

    let batch = project_over_parquet("query-engine-native-integer-casts", input, exprs, output_schema).unwrap();

    assert_eq!(int32_array(&batch, 0).iter().collect::<Vec<_>>(), [Some(i32::MAX), Some(i32::MIN), Some(0), None]);
    assert_eq!(int64_array(&batch, 1).iter().collect::<Vec<_>>(), [Some(0), Some(i64::MAX), Some(7), None]);
    let narrow = batch.column(2).as_any().downcast_ref::<UInt8Array>().unwrap();
    assert_eq!(narrow.iter().collect::<Vec<_>>(), [Some(255), Some(0), Some(7), None]);
    let low = batch.column(3).as_any().downcast_ref::<StringArray>().unwrap();
    assert_eq!(low.iter().collect::<Vec<_>>(), [Some("-9223372036854775808"), Some("-1"), Some("0"), None]);
    let top = batch.column(4).as_any().downcast_ref::<StringArray>().unwrap();
    assert_eq!(top.iter().collect::<Vec<_>>(), [Some("18446744073709551615"), Some("0"), Some("1"), None]);
}

fn native_query_range_checked_casts_fail_the_query_outside_the_target_range() {
    let input = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("wide", DataType::Int64, true),
            Field::new("negative", DataType::Int64, true),
            Field::new("score", DataType::Float64, true),
        ])),
        vec![
            Arc::new(Int64Array::from(vec![Some(1), Some(i64::from(i32::MAX) + 1)])),
            Arc::new(Int64Array::from(vec![Some(1), Some(-1)])),
            Arc::new(Float64Array::from(vec![Some(1.0), Some(f64::NAN)])),
        ],
    )
    .unwrap();
    for (name, column, target) in [
        ("int64-to-int32", "wide", DataType::Int32),
        ("int64-to-uint64", "negative", DataType::UInt64),
        ("float64-nan-to-int32", "score", DataType::Int32),
    ] {
        let output_schema = Arc::new(Schema::new(vec![Field::new("out", target.clone(), true)]));
        let error = project_over_parquet(
            &format!("query-engine-native-cast-failure-{name}"),
            input.clone(),
            vec![Expr::cast(Expr::column(column), target)],
            output_schema,
        )
        .expect_err(name);
        assert_cudf_input_failure(&error, "ValueOutOfRange");
    }
}

fn native_query_integer_division_fails_on_zero_divisors_and_overflow_only_in_valid_rows() {
    let batch_with = |a: Vec<Option<i64>>, b: Vec<Option<i64>>| {
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, true), Field::new("b", DataType::Int64, true)])),
            vec![Arc::new(Int64Array::from(a)), Arc::new(Int64Array::from(b))],
        )
        .unwrap()
    };
    let output_schema = || Arc::new(Schema::new(vec![Field::new("out", DataType::Int64, true)]));
    let divide = |op| vec![Expr::binary(Expr::column("a"), op, Expr::column("b"))];

    for (name, op) in [("division", BinaryOp::Divide), ("remainder", BinaryOp::Modulo)] {
        let error = project_over_parquet(
            &format!("query-engine-native-{name}-by-zero"),
            batch_with(vec![Some(1), Some(2)], vec![Some(1), Some(0)]),
            divide(op),
            output_schema(),
        )
        .expect_err(name);
        assert_cudf_input_failure(&error, "DivisionByZero");
    }

    let overflow = project_over_parquet(
        "query-engine-native-division-overflow",
        batch_with(vec![Some(i64::MIN)], vec![Some(-1)]),
        divide(BinaryOp::Divide),
        output_schema(),
    )
    .unwrap_err();
    assert_cudf_input_failure(&overflow, "ValueOutOfRange");

    let wrapped = project_over_parquet(
        "query-engine-native-remainder-minus-one",
        batch_with(vec![Some(i64::MIN), Some(5)], vec![Some(-1), Some(-1)]),
        divide(BinaryOp::Modulo),
        output_schema(),
    )
    .unwrap();
    assert_eq!(int64_array(&wrapped, 0).iter().collect::<Vec<_>>(), [Some(0), Some(0)]);

    let null_rows = project_over_parquet(
        "query-engine-native-zero-divisor-in-null-row",
        batch_with(vec![None, Some(8)], vec![Some(0), Some(2)]),
        divide(BinaryOp::Divide),
        output_schema(),
    )
    .unwrap();
    assert_eq!(int64_array(&null_rows, 0).iter().collect::<Vec<_>>(), [None, Some(4)]);
}

fn native_query_floor_and_ceil_match_ieee_rounding() {
    let wide = [Some(1.5), Some(-1.5), Some(f64::NAN), Some(f64::INFINITY), Some(-0.0), None, Some(2.0)];
    let narrow = wide.map(|value| value.map(|value| value as f32));
    let input = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("wide", DataType::Float64, true),
            Field::new("narrow", DataType::Float32, true),
        ])),
        vec![Arc::new(Float64Array::from(wide.to_vec())), Arc::new(Float32Array::from(narrow.to_vec()))],
    )
    .unwrap();
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("floor_wide", DataType::Float64, true),
        Field::new("ceil_wide", DataType::Float64, true),
        Field::new("floor_narrow", DataType::Float32, true),
        Field::new("ceil_narrow", DataType::Float32, true),
    ]));
    let exprs = vec![
        Expr::unary_math(UnaryMathOp::Floor, Expr::column("wide"), DataType::Float64),
        Expr::unary_math(UnaryMathOp::Ceil, Expr::column("wide"), DataType::Float64),
        Expr::unary_math(UnaryMathOp::Floor, Expr::column("narrow"), DataType::Float32),
        Expr::unary_math(UnaryMathOp::Ceil, Expr::column("narrow"), DataType::Float32),
    ];

    let batch = project_over_parquet("query-engine-native-floor-ceil", input, exprs, output_schema).unwrap();

    let bits = |values: Vec<Option<f64>>| values.into_iter().map(|value| value.map(f64::to_bits)).collect::<Vec<_>>();
    let nan_bits = f64::NAN.to_bits();
    let normalized = |values: Vec<Option<u64>>| {
        values
            .into_iter()
            .map(|value| value.map(|bits| if f64::from_bits(bits).is_nan() { nan_bits } else { bits }))
            .collect::<Vec<_>>()
    };
    let actual_wide = |index: usize| {
        normalized(bits(batch.column(index).as_any().downcast_ref::<Float64Array>().unwrap().iter().collect()))
    };
    assert_eq!(actual_wide(0), normalized(bits(wide.iter().map(|value| value.map(f64::floor)).collect())));
    assert_eq!(actual_wide(1), normalized(bits(wide.iter().map(|value| value.map(f64::ceil)).collect())));
    let actual_narrow = |index: usize| {
        normalized(bits(
            batch
                .column(index)
                .as_any()
                .downcast_ref::<Float32Array>()
                .unwrap()
                .iter()
                .map(|value| value.map(f64::from))
                .collect(),
        ))
    };
    assert_eq!(
        actual_narrow(2),
        normalized(bits(narrow.iter().map(|value| value.map(|value| f64::from(value.floor()))).collect()))
    );
    assert_eq!(
        actual_narrow(3),
        normalized(bits(narrow.iter().map(|value| value.map(|value| f64::from(value.ceil()))).collect()))
    );
}

fn decimal_cast_input() -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("amount", DataType::Decimal128(12, 4), true),
            Field::new("whole", DataType::Int64, true),
            Field::new("score", DataType::Float64, true),
        ])),
        vec![
            Arc::new(
                Decimal128Array::from(vec![
                    Some(1_249),
                    Some(1_250),
                    Some(-1_250),
                    Some(5),
                    Some(-5),
                    Some(999_999_999_999),
                    None,
                ])
                .with_precision_and_scale(12, 4)
                .unwrap(),
            ),
            Arc::new(Int64Array::from(vec![
                Some(1),
                Some(-12_345),
                Some(0),
                Some(999),
                Some(-999),
                Some(999_999_999_999_999),
                None,
            ])),
            Arc::new(Float64Array::from(vec![
                Some(0.125),
                Some(-0.125),
                Some(2.5),
                Some(-2.5),
                Some(1234.5678),
                Some(0.285),
                None,
            ])),
        ],
    )
    .unwrap()
}

fn native_query_decimal_casts_fail_the_query_outside_the_target_precision() {
    let nan_score = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("score", DataType::Float64, true)])),
        vec![Arc::new(Float64Array::from(vec![Some(1.5), Some(f64::NAN)]))],
    )
    .unwrap();
    for (name, input, column, target) in [
        // 99_999_999.9999 rounds to 100_000_000.00, which needs 11 digits.
        ("decimal-to-narrow-precision", decimal_cast_input(), "amount", DataType::Decimal128(10, 2)),
        ("integer-to-narrow-precision", decimal_cast_input(), "whole", DataType::Decimal128(10, 2)),
        ("float-to-narrow-precision", decimal_cast_input(), "score", DataType::Decimal128(5, 2)),
        ("float-nan", nan_score, "score", DataType::Decimal128(10, 2)),
    ] {
        let output_schema = Arc::new(Schema::new(vec![Field::new("out", target.clone(), true)]));
        let error = project_over_parquet(
            &format!("query-engine-native-decimal-cast-failure-{name}"),
            input,
            vec![Expr::cast(Expr::column(column), target)],
            output_schema,
        )
        .expect_err(name);
        assert_cudf_input_failure(&error, "ValueOutOfRange");
    }
}

#[test]
fn scalar_expressions_cases_1() {
    run_named_cases(&[
        (
            "projection_validation::scalar_expressions::native_query_decimal_casts_fail_the_query_outside_the_target_precision",
            native_query_decimal_casts_fail_the_query_outside_the_target_precision,
        ),
        (
            "projection_validation::scalar_expressions::native_query_floor_and_ceil_match_ieee_rounding",
            native_query_floor_and_ceil_match_ieee_rounding,
        ),
        (
            "projection_validation::scalar_expressions::native_query_integer_division_fails_on_zero_divisors_and_overflow_only_in_valid_rows",
            native_query_integer_division_fails_on_zero_divisors_and_overflow_only_in_valid_rows,
        ),
        (
            "projection_validation::scalar_expressions::native_query_casts_integers_between_widths_and_formats_them_as_text",
            native_query_casts_integers_between_widths_and_formats_them_as_text,
        ),
        (
            "projection_validation::scalar_expressions::native_query_executes_round_scalar_expression",
            native_query_executes_round_scalar_expression,
        ),
    ]);
}
