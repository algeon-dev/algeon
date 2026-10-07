use std::{fs::File, sync::Arc};

use arrow::compute::{CastOptions, cast_with_options, kernels::numeric::sub};
use query_planning::{
    error::QueryError,
    expr::{CaseWhen, DatePart, UnaryMathOp},
};

use super::*;

#[test]
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

#[test]
fn native_query_executes_scalar_expression_families() {
    let dir = unique_temp_dir("query-engine-native-execution-scalar_expression-scalars");
    let path = dir.join("tpch-like.parquet");
    let context = backend_context();
    let runtime = Arc::new(context.runtime().unwrap());
    let input_schema = write_scalar_expression_parquet(&runtime, &path);
    let projected_schema = Arc::new(Schema::new(vec![
        Field::new("orderkey", DataType::Int64, false),
        Field::new("order_mod", DataType::Int64, false),
        Field::new("ship_year", DataType::Int32, false),
        Field::new("ship_month", DataType::Int32, false),
        Field::new("ship_day", DataType::Int32, false),
        Field::new("event_ts_minus_interval", DataType::Timestamp(TimeUnit::Nanosecond, None), false),
        Field::new("status_prefix", DataType::Utf8, true),
        Field::new("status_label", DataType::Utf8, false),
        Field::new("status_known", DataType::Boolean, false),
        Field::new("status_is_null", DataType::Boolean, false),
        Field::new("status_not_f", DataType::Boolean, true),
        Field::new("status_like_o", DataType::Boolean, true),
        Field::new("status_not_like_f", DataType::Boolean, true),
        Field::new("status_contains_o", DataType::Boolean, true),
        Field::new("status_starts_o", DataType::Boolean, true),
        Field::new("status_ends_o", DataType::Boolean, true),
        Field::new("abs_order_delta", DataType::Int64, false),
        Field::new("sqrt_orderkey", DataType::Float64, false),
        Field::new("status_distinct_from_o", DataType::Boolean, false),
        Field::new("status_not_distinct_from_o", DataType::Boolean, false),
    ]));
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["orderkey".to_string(), "shipdate".to_string(), "event_ts".to_string(), "status".to_string()],
    );
    let filter = Expr::binary(
        Expr::in_list(
            Expr::column("orderkey"),
            vec![
                Expr::literal(LiteralValue::Int64(1)),
                Expr::literal(LiteralValue::Int64(3)),
                Expr::literal(LiteralValue::Int64(4)),
            ],
        ),
        BinaryOp::And,
        Expr::is_not_null(Expr::column("shipdate")),
    );
    let projection = vec![
        Expr::column("orderkey"),
        Expr::binary(Expr::column("orderkey"), BinaryOp::Modulo, Expr::literal(LiteralValue::Int64(2)))
            .alias("order_mod"),
        Expr::extract(DatePart::Year, Expr::column("shipdate")).alias("ship_year"),
        Expr::extract(DatePart::Month, Expr::column("shipdate")).alias("ship_month"),
        Expr::extract(DatePart::Day, Expr::column("shipdate")).alias("ship_day"),
        Expr::binary(
            Expr::column("event_ts"),
            BinaryOp::Subtract,
            Expr::literal(LiteralValue::IntervalMonthDayNano { months: 0, days: 0, nanoseconds: 50 }),
        )
        .alias("event_ts_minus_interval"),
        Expr::substring(Expr::column("status"), 1, Some(1)).alias("status_prefix"),
        Expr::case(
            vec![CaseWhen::new(
                Expr::binary(Expr::column("status"), BinaryOp::Eq, Expr::literal(LiteralValue::Utf8("O".to_string()))),
                Expr::literal(LiteralValue::Utf8("open".to_string())),
            )],
            Expr::literal(LiteralValue::Utf8("other".to_string())),
        )
        .alias("status_label"),
        Expr::is_not_null(Expr::column("status")).alias("status_known"),
        Expr::is_null(Expr::column("status")).alias("status_is_null"),
        Expr::not_in_list(Expr::column("status"), vec![Expr::literal(LiteralValue::Utf8("F".to_string()))])
            .alias("status_not_f"),
        Expr::like(Expr::column("status"), "O%").alias("status_like_o"),
        Expr::not_like(Expr::column("status"), "F%").alias("status_not_like_f"),
        Expr::string_contains(Expr::column("status"), "O").alias("status_contains_o"),
        Expr::starts_with(Expr::column("status"), "O").alias("status_starts_o"),
        Expr::ends_with(Expr::column("status"), "O").alias("status_ends_o"),
        Expr::abs(Expr::binary(Expr::column("orderkey"), BinaryOp::Subtract, Expr::literal(LiteralValue::Int64(3))))
            .alias("abs_order_delta"),
        Expr::unary_math(UnaryMathOp::Sqrt, Expr::column("orderkey"), DataType::Float64).alias("sqrt_orderkey"),
        Expr::is_distinct_from(Expr::column("status"), Expr::literal(LiteralValue::Utf8("O".to_string())))
            .alias("status_distinct_from_o"),
        Expr::is_not_distinct_from(Expr::column("status"), Expr::literal(LiteralValue::Utf8("O".to_string())))
            .alias("status_not_distinct_from_o"),
    ];

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let filter = builder.add_filter(source, filter).unwrap();
    let projection = builder.add_projection(filter, projection_outputs(projection, projected_schema)).unwrap();
    let plan = builder.finish_with_host_sink(projection).unwrap();

    let result = execute_query_plan(&context, &plan).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 3);
    assert_eq!(int64_array(batch, 0).values().as_ref(), &[1, 3, 4]);
    assert_eq!(int64_array(batch, 1).values().as_ref(), &[1, 1, 0]);
    assert_eq!(int32_array(batch, 2).values().as_ref(), &[2020, 2020, 2020]);
    assert_eq!(int32_array(batch, 3).values().as_ref(), &[1, 3, 3]);
    assert_eq!(int32_array(batch, 4).values().as_ref(), &[1, 1, 15]);
    assert_eq!(timestamp_ns_array(batch, 5).values().as_ref(), &[50, 250, 350]);
    assert_eq!(string_array(batch, 6).value(0), "O");
    assert_eq!(string_array(batch, 6).value(1), "O");
    assert!(string_array(batch, 6).is_null(2));
    assert_eq!(string_array(batch, 7).value(0), "open");
    assert_eq!(string_array(batch, 7).value(1), "open");
    assert_eq!(string_array(batch, 7).value(2), "other");
    assert_eq!(boolean_array(batch, 8).values().iter().collect::<Vec<_>>(), vec![true, true, false]);
    assert_eq!(boolean_array(batch, 9).values().iter().collect::<Vec<_>>(), vec![false, false, true]);
    assert!(boolean_array(batch, 10).value(0));
    assert!(boolean_array(batch, 10).value(1));
    assert!(boolean_array(batch, 10).is_null(2));
    assert!(boolean_array(batch, 11).value(0));
    assert!(boolean_array(batch, 11).value(1));
    assert!(boolean_array(batch, 11).is_null(2));
    assert!(boolean_array(batch, 12).value(0));
    assert!(boolean_array(batch, 12).value(1));
    assert!(boolean_array(batch, 12).is_null(2));
    assert!(boolean_array(batch, 13).value(0));
    assert!(boolean_array(batch, 13).value(1));
    assert!(boolean_array(batch, 13).is_null(2));
    assert!(boolean_array(batch, 14).value(0));
    assert!(boolean_array(batch, 14).value(1));
    assert!(boolean_array(batch, 14).is_null(2));
    assert!(boolean_array(batch, 15).value(0));
    assert!(boolean_array(batch, 15).value(1));
    assert!(boolean_array(batch, 15).is_null(2));
    assert_eq!(int64_array(batch, 16).values().as_ref(), &[2, 0, 1]);
    assert_eq!(float64_array(batch, 17).value(0), 1.0);
    assert!((float64_array(batch, 17).value(1) - 3.0_f64.sqrt()).abs() < 1e-12);
    assert_eq!(float64_array(batch, 17).value(2), 2.0);
    assert!(!boolean_array(batch, 18).value(0));
    assert!(!boolean_array(batch, 18).value(1));
    assert!(boolean_array(batch, 18).value(2));
    assert!(boolean_array(batch, 19).value(0));
    assert!(boolean_array(batch, 19).value(1));
    assert!(!boolean_array(batch, 19).value(2));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
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

#[test]
fn native_query_executes_float64_to_int64_cast_with_datafusion_truncation_semantics() {
    let dir = unique_temp_dir("query-engine-native-execution-f64-i64-cast");
    let path = dir.join("input.parquet");
    let context = backend_context();
    let input_schema = Arc::new(Schema::new(vec![Field::new("score", DataType::Float64, true)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&input_schema),
        vec![Arc::new(Float64Array::from(vec![Some(1.9), Some(-1.9), None, Some(0.0)]))],
    )
    .unwrap();
    let file = File::create(&path).unwrap();
    let mut writer = ArrowWriter::try_new(file, Arc::clone(&input_schema), None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();

    let output_schema = Arc::new(Schema::new(vec![Field::new("score_i", DataType::Int64, true)]));
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["score".to_string()],
    );
    let projection = vec![Expr::cast(Expr::column("score"), DataType::Int64).alias("score_i")];

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let projection = builder.add_projection(source, projection_outputs(projection, output_schema)).unwrap();
    let plan = builder.finish_with_host_sink(projection).unwrap();

    let result = execute_query_plan(&context, &plan).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    let values = int64_array(batch, 0);
    assert_eq!(values.value(0), 1);
    assert_eq!(values.value(1), -1);
    assert!(values.is_null(2));
    assert_eq!(values.value(3), 0);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_query_rejects_invalid_float64_to_int64_cast_inputs() {
    let dir = unique_temp_dir("query-engine-native-execution-f64-i64-cast-invalid");
    let path = dir.join("input.parquet");
    let context = backend_context();
    let input_schema = Arc::new(Schema::new(vec![Field::new("score", DataType::Float64, true)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&input_schema),
        vec![Arc::new(Float64Array::from(vec![Some(f64::NAN), Some(f64::INFINITY), Some(1.0e40)]))],
    )
    .unwrap();
    let file = File::create(&path).unwrap();
    let mut writer = ArrowWriter::try_new(file, Arc::clone(&input_schema), None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();

    let output_schema = Arc::new(Schema::new(vec![Field::new("score_i", DataType::Int64, true)]));
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["score".to_string()],
    );
    let projection = vec![Expr::cast(Expr::column("score"), DataType::Int64).alias("score_i")];

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let projection = builder.add_projection(source, projection_outputs(projection, output_schema)).unwrap();
    let plan = builder.finish_with_host_sink(projection).unwrap();

    let error = match execute_query_plan(&context, &plan) {
        Ok(_) => panic!("invalid Float64 -> Int64 casts must fail"),
        Err(error) => error,
    };

    assert_eq!(error.code(), QueryErrorCode::CudfInvalidInput, "got {error:?}");
    assert_eq!(error.fact_value("cudf_code"), Some("ValueOutOfRange"));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
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
    let dir = unique_temp_dir("query-engine-native-execution-scalar_expression-type-reject");
    let path = dir.join("tpch-like.parquet");
    let context = backend_context();
    let runtime = Arc::new(context.runtime().unwrap());
    let input_schema = write_scalar_expression_parquet(&runtime, &path);
    let output_schema = Arc::new(Schema::new(vec![Field::new("bad_like", DataType::Boolean, true)]));
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
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

    fs::remove_dir_all(dir).unwrap();
}

#[test]
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
    let dir = unique_temp_dir("query-engine-native-execution-date-trunc-output-unit-reject");
    let path = dir.join("tpch-like.parquet");
    let context = backend_context();
    let runtime = Arc::new(context.runtime().unwrap());
    let input_schema = write_scalar_expression_parquet(&runtime, &path);
    let output_schema =
        Arc::new(Schema::new(vec![Field::new("bad_trunc", DataType::Timestamp(TimeUnit::Millisecond, None), false)]));
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

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_query_rejects_implicit_decimal_float_arithmetic_before_cudf_execution() {
    let dir = unique_temp_dir("query-engine-native-execution-decimal-float-implicit-reject");
    let path = dir.join("tpch-like.parquet");
    let context = backend_context();
    let runtime = Arc::new(context.runtime().unwrap());
    let input_schema = write_tpch_like_types_parquet(&runtime, &path);
    let output_schema = Arc::new(Schema::new(vec![Field::new("bad_amount", DataType::Decimal128(15, 2), true)]));
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
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

    fs::remove_dir_all(dir).unwrap();
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

#[test]
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

#[test]
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
    let dir = unique_temp_dir("query-engine-native-execution-scalar_expression-source-column-reject");
    let path = dir.join("tpch-like.parquet");
    let context = backend_context();
    let runtime = Arc::new(context.runtime().unwrap());
    let input_schema = write_scalar_expression_parquet(&runtime, &path);
    let output_schema = Arc::new(Schema::new(vec![Field::new("status_label", DataType::Utf8, true)]));
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
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

    fs::remove_dir_all(dir).unwrap();
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

#[test]
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

#[test]
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

#[test]
fn native_query_integer_division_and_remainder_follow_arrow_row_semantics() {
    let dividends = [Some(7), Some(-7), Some(0), None, Some(9), Some(i64::MIN)];
    let divisors = [Some(2), Some(2), Some(5), Some(0), None, Some(3)];
    let input = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, true), Field::new("b", DataType::Int64, true)])),
        vec![Arc::new(Int64Array::from(dividends.to_vec())), Arc::new(Int64Array::from(divisors.to_vec()))],
    )
    .unwrap();
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("quotient", DataType::Int64, true),
        Field::new("remainder", DataType::Int64, true),
        Field::new("quotient_20", DataType::Int64, true),
        Field::new("remainder_20", DataType::Int64, true),
    ]));
    let twenty = || Expr::literal(LiteralValue::Int64(20));
    let exprs = vec![
        Expr::binary(Expr::column("a"), BinaryOp::Divide, Expr::column("b")),
        Expr::binary(Expr::column("a"), BinaryOp::Modulo, Expr::column("b")),
        Expr::binary(Expr::column("a"), BinaryOp::Divide, twenty()),
        Expr::binary(Expr::column("a"), BinaryOp::Modulo, twenty()),
    ];

    let batch = project_over_parquet("query-engine-native-integer-division", input, exprs, output_schema).unwrap();

    let expect = |op: fn(i64, i64) -> i64| -> Vec<Option<i64>> {
        dividends.iter().zip(divisors).map(|(a, b)| a.zip(b).map(|(a, b)| op(a, b))).collect()
    };
    assert_eq!(int64_array(&batch, 0).iter().collect::<Vec<_>>(), expect(|a, b| a / b));
    assert_eq!(int64_array(&batch, 1).iter().collect::<Vec<_>>(), expect(|a, b| a % b));
    let by_twenty =
        |op: fn(i64, i64) -> i64| -> Vec<Option<i64>> { dividends.iter().map(|a| a.map(|a| op(a, 20))).collect() };
    assert_eq!(int64_array(&batch, 2).iter().collect::<Vec<_>>(), by_twenty(|a, b| a / b));
    assert_eq!(int64_array(&batch, 3).iter().collect::<Vec<_>>(), by_twenty(|a, b| a % b));
}

#[test]
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

#[test]
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

#[test]
fn native_query_casts_into_decimal_follow_arrow_rescaling_and_rounding() {
    let input = decimal_cast_input();
    let casts = [
        ("amount", DataType::Decimal128(12, 2)),
        ("amount", DataType::Decimal128(20, 8)),
        ("whole", DataType::Decimal128(20, 2)),
        ("score", DataType::Decimal128(10, 2)),
        ("score", DataType::Decimal128(38, 0)),
    ];
    let output_schema = Arc::new(Schema::new(
        casts
            .iter()
            .enumerate()
            .map(|(index, (_, target))| Field::new(format!("out_{index}"), target.clone(), true))
            .collect::<Vec<_>>(),
    ));
    let exprs = casts.iter().map(|(column, target)| Expr::cast(Expr::column(*column), target.clone())).collect();

    let batch = project_over_parquet("query-engine-native-decimal-casts", input.clone(), exprs, output_schema).unwrap();

    let arrow_options = CastOptions { safe: false, ..CastOptions::default() };
    for (index, (column, target)) in casts.iter().enumerate() {
        let source = input.column_by_name(column).unwrap();
        let expected = cast_with_options(source, target, &arrow_options).unwrap();
        assert_eq!(batch.column(index).as_ref(), expected.as_ref(), "{column} as {target:?}");
    }
}

#[test]
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
fn native_query_casts_timestamp_and_duration_counts_to_int64_like_arrow() {
    let input_ms = [Some(1_700_000_000_123), Some(-86_400_001), Some(0), None, Some(5), Some(-3)];
    let later_ms = [Some(1_700_000_001_623), Some(3_600_000), None, Some(9), Some(4), Some(-3)];
    // The sub-unit offsets keep each unit's count distinct from a rescaled one.
    let scaled = |values: &[Option<i64>], factor: i64, offset: i64| -> Vec<Option<i64>> {
        values.iter().map(|value| value.map(|value| value * factor + offset)).collect()
    };
    let timestamps = |unit: TimeUnit, values: Vec<Option<i64>>| -> ArrayRef {
        cast_with_options(&Int64Array::from(values), &DataType::Timestamp(unit, None), &CastOptions::default()).unwrap()
    };
    let columns = [
        ("ts_ms", timestamps(TimeUnit::Millisecond, input_ms.to_vec())),
        ("later_ms", timestamps(TimeUnit::Millisecond, later_ms.to_vec())),
        ("ts_us", timestamps(TimeUnit::Microsecond, scaled(&input_ms, 1_000, 7))),
        ("later_us", timestamps(TimeUnit::Microsecond, scaled(&later_ms, 1_000, 3))),
        ("ts_ns", timestamps(TimeUnit::Nanosecond, scaled(&input_ms, 1_000_000, 7))),
        ("later_ns", timestamps(TimeUnit::Nanosecond, scaled(&later_ms, 1_000_000, 3))),
    ];
    let input = RecordBatch::try_new(
        Arc::new(Schema::new(
            columns.iter().map(|(name, array)| Field::new(*name, array.data_type().clone(), true)).collect::<Vec<_>>(),
        )),
        columns.iter().map(|(_, array)| Arc::clone(array)).collect(),
    )
    .unwrap();

    let elapsed = |unit: &str| {
        Expr::binary(Expr::column(format!("later_{unit}")), BinaryOp::Subtract, Expr::column(format!("ts_{unit}")))
    };
    let mut exprs = Vec::new();
    for unit in ["ms", "us", "ns"] {
        exprs.push(Expr::cast(Expr::column(format!("ts_{unit}")), DataType::Int64));
        exprs.push(Expr::cast(elapsed(unit), DataType::Int64));
    }
    exprs.push(elapsed("ms"));
    let mut output_fields =
        (0..6).map(|index| Field::new(format!("count_{index}"), DataType::Int64, true)).collect::<Vec<_>>();
    output_fields.push(Field::new("elapsed", DataType::Duration(TimeUnit::Millisecond), true));

    let batch = project_over_parquet(
        "query-engine-native-temporal-count-casts",
        input.clone(),
        exprs,
        Arc::new(Schema::new(output_fields)),
    )
    .unwrap();

    let arrow_options = CastOptions { safe: false, ..CastOptions::default() };
    let column = |name: &str| input.column_by_name(name).unwrap();
    for (index, unit) in ["ms", "us", "ns"].into_iter().enumerate() {
        let ts = column(&format!("ts_{unit}"));
        let difference = sub(column(&format!("later_{unit}")), ts).unwrap();
        for (output, source) in [(2 * index, Arc::clone(ts)), (2 * index + 1, difference)] {
            let expected = cast_with_options(&source, &DataType::Int64, &arrow_options).unwrap();
            assert_eq!(batch.column(output).as_ref(), expected.as_ref(), "{unit} output {output}");
        }
    }
    let expected_elapsed = sub(column("later_ms"), column("ts_ms")).unwrap();
    assert_eq!(batch.column(6).as_ref(), expected_elapsed.as_ref());
}

fn nanosecond_timestamp_input(columns: [(&str, Vec<Option<i64>>); 3]) -> RecordBatch {
    let timestamp = DataType::Timestamp(TimeUnit::Nanosecond, None);
    RecordBatch::try_new(
        Arc::new(Schema::new(
            columns.iter().map(|(name, _)| Field::new(*name, timestamp.clone(), true)).collect::<Vec<_>>(),
        )),
        columns
            .into_iter()
            .map(|(_, values)| {
                cast_with_options(&Int64Array::from(values), &timestamp, &CastOptions::default()).unwrap()
            })
            .collect(),
    )
    .unwrap()
}

#[test]
fn native_query_timestamp_arithmetic_fails_the_query_where_arrow_checks_the_result() {
    // base + (x - y): the elapsed duration, then the shifted timestamp.
    let elapsed = || Expr::binary(Expr::column("x"), BinaryOp::Subtract, Expr::column("y"));
    let exprs = || vec![elapsed(), Expr::binary(Expr::column("base"), BinaryOp::Add, elapsed())];
    let output_schema = || {
        Arc::new(Schema::new(vec![
            Field::new("elapsed", DataType::Duration(TimeUnit::Nanosecond), true),
            Field::new("shifted", DataType::Timestamp(TimeUnit::Nanosecond, None), true),
        ]))
    };
    let counts = |batch: &RecordBatch, index: usize| {
        let counts = cast_with_options(batch.column(index), &DataType::Int64, &CastOptions::default()).unwrap();
        counts.as_any().downcast_ref::<Int64Array>().unwrap().iter().collect::<Vec<_>>()
    };

    // Row 1 would overflow under a NULL `y`, row 2 under a NULL `base`, and row 3 lands exactly on the limit.
    let accepted = nanosecond_timestamp_input([
        ("base", vec![Some(10), Some(i64::MAX - 1), None, Some(i64::MAX - 1)]),
        ("x", vec![Some(5), Some(5), Some(i64::MAX), Some(1)]),
        ("y", vec![Some(2), None, Some(0), Some(0)]),
    ]);
    let batch =
        project_over_parquet("query-engine-native-timestamp-arithmetic", accepted, exprs(), output_schema()).unwrap();
    assert_eq!(counts(&batch, 0), [Some(3), None, Some(i64::MAX), Some(1)]);
    assert_eq!(counts(&batch, 1), [Some(13), None, None, Some(i64::MAX)]);

    for (name, columns) in [
        ("addition", [("base", vec![Some(i64::MAX - 1)]), ("x", vec![Some(5)]), ("y", vec![Some(0)])]),
        ("subtraction", [("base", vec![Some(0)]), ("x", vec![Some(i64::MAX)]), ("y", vec![Some(i64::MIN)])]),
    ] {
        let error = project_over_parquet(
            &format!("query-engine-native-timestamp-arithmetic-failure-{name}"),
            nanosecond_timestamp_input(columns),
            exprs(),
            output_schema(),
        )
        .expect_err(name);
        assert_cudf_input_failure(&error, "ValueOutOfRange");
    }
}
