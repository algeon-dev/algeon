use super::*;

#[test]
fn native_query_executes_scalar_expression_families() {
    let dir = unique_temp_dir("nexus-query-engine-native-execution-scalar_expression-scalars");
    let path = dir.join("tpch-like.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
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
        Expr::extract(nexus_query_engine::DatePart::Year, Expr::column("shipdate")).alias("ship_year"),
        Expr::extract(nexus_query_engine::DatePart::Month, Expr::column("shipdate")).alias("ship_month"),
        Expr::extract(nexus_query_engine::DatePart::Day, Expr::column("shipdate")).alias("ship_day"),
        Expr::binary(
            Expr::column("event_ts"),
            BinaryOp::Subtract,
            Expr::literal(LiteralValue::IntervalMonthDayNano { months: 0, days: 0, nanoseconds: 50 }),
        )
        .alias("event_ts_minus_interval"),
        Expr::substring(Expr::column("status"), 1, Some(1)).alias("status_prefix"),
        Expr::case(
            vec![nexus_query_engine::CaseWhen::new(
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
        Expr::abs(Expr::binary(Expr::column("orderkey"), BinaryOp::Subtract, Expr::literal(LiteralValue::Int64(3))))
            .alias("abs_order_delta"),
        Expr::sqrt(Expr::column("orderkey"), DataType::Float64).alias("sqrt_orderkey"),
        Expr::is_distinct_from(Expr::column("status"), Expr::literal(LiteralValue::Utf8("O".to_string())))
            .alias("status_distinct_from_o"),
        Expr::is_not_distinct_from(Expr::column("status"), Expr::literal(LiteralValue::Utf8("O".to_string())))
            .alias("status_not_distinct_from_o"),
    ];

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let filter = builder.add_filter(source, filter, Arc::clone(&input_schema));
    let projection = builder.add_projection(filter, projection_outputs(projection, projected_schema));
    let plan = builder.finish_with_host_sink(projection);

    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime));
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
    assert_eq!(int64_array(batch, 13).values().as_ref(), &[2, 0, 1]);
    assert_eq!(float64_array(batch, 14).value(0), 1.0);
    assert!((float64_array(batch, 14).value(1) - 3.0_f64.sqrt()).abs() < 1e-12);
    assert_eq!(float64_array(batch, 14).value(2), 2.0);
    assert!(!boolean_array(batch, 15).value(0));
    assert!(!boolean_array(batch, 15).value(1));
    assert!(boolean_array(batch, 15).value(2));
    assert!(boolean_array(batch, 16).value(0));
    assert!(boolean_array(batch, 16).value(1));
    assert!(!boolean_array(batch, 16).value(2));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_query_executes_bitwise_scalar_expressions() {
    let dir = unique_temp_dir("nexus-query-engine-native-execution-bitwise-scalars");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
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
    let projection = builder.add_projection(source, projection_outputs(projection, output_schema));
    let plan = builder.finish_with_host_sink(projection);

    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime));
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
    let dir = unique_temp_dir("nexus-query-engine-native-execution-f64-i64-cast");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
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
    let projection = builder.add_projection(source, projection_outputs(projection, output_schema));
    let plan = builder.finish_with_host_sink(projection);

    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime));
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
    let dir = unique_temp_dir("nexus-query-engine-native-execution-f64-i64-cast-invalid");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
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
    let projection = builder.add_projection(source, projection_outputs(projection, output_schema));
    let plan = builder.finish_with_host_sink(projection);

    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime));
    let error = match execute_query_plan(&context, &plan) {
        Ok(_) => panic!("invalid Float64 -> Int64 casts must fail"),
        Err(error) => error,
    };

    assert!(
        matches!(
            error.code(),
            QueryErrorCode::CudfInvalidInput
                | QueryErrorCode::CudfUnsupported
                | QueryErrorCode::CudfPersistentFailure
                | QueryErrorCode::CudfInvariantViolation
        ),
        "invalid Float64 -> Int64 cast should be rejected by the native cast contract, got {error:?}",
    );

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_query_executes_regexp_replace_projection_with_backrefs() {
    let dir = unique_temp_dir("nexus-query-engine-native-execution-regexp-replace");
    let path = dir.join("referers.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
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
    let projection = builder.add_projection(source, projection_outputs(projection, output_schema));
    let plan = builder.finish_with_host_sink(projection);

    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime));
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
    let dir = unique_temp_dir("nexus-query-engine-native-execution-scalar_expression-type-reject");
    let path = dir.join("tpch-like.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_scalar_expression_parquet(&runtime, &path);
    let output_schema = Arc::new(Schema::new(vec![Field::new("bad_like", DataType::Boolean, true)]));
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["orderkey".to_string()],
    );

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let projection = builder.add_projection(
        source,
        projection_outputs(vec![Expr::like(Expr::column("orderkey"), "1%").alias("bad_like")], output_schema),
    );
    let plan = builder.finish_with_host_sink(projection);
    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime));

    let error = match execute_query_plan(&context, &plan) {
        Ok(_) => panic!("native plan validation should reject LIKE over an integral input before cuDF execution"),
        Err(error) => error,
    };
    assert_eq!(error.code(), QueryErrorCode::UnsupportedType);
    assert_eq!(error.operation(), "nexus_query_engine.plan.validate_expr");
    assert_eq!(error.fact_value("context"), Some("LIKE input"));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_query_rejects_implicit_decimal_float_arithmetic_before_cudf_execution() {
    let dir = unique_temp_dir("nexus-query-engine-native-execution-decimal-float-implicit-reject");
    let path = dir.join("tpch-like.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_tpch_like_types_parquet(&runtime, &path);
    let output_schema = Arc::new(Schema::new(vec![Field::new("bad_amount", DataType::Decimal128(15, 2), true)]));
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["amount".to_string()],
    );

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let projection = builder.add_projection(
        source,
        projection_outputs(
            vec![
                Expr::binary(Expr::column("amount"), BinaryOp::Add, Expr::literal(LiteralValue::Float64(1.0)))
                    .alias("bad_amount"),
            ],
            output_schema,
        ),
    );
    let plan = builder.finish_with_host_sink(projection);
    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime));

    let error = match execute_query_plan(&context, &plan) {
        Ok(_) => panic!("native plan validation should reject implicit Decimal128/Float64 arithmetic"),
        Err(error) => error,
    };
    assert_eq!(error.code(), QueryErrorCode::UnsupportedType);
    assert_eq!(error.operation(), "nexus_query_engine.plan.validate_expr");
    assert_eq!(error.fact_value("context"), Some("binary arithmetic operands"));
    assert_eq!(error.fact_value("left_type"), Some("Decimal128(15, 2)"));
    assert_eq!(error.fact_value("right_type"), Some("Float64"));
    assert!(error.fact_value("path").is_none());

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_query_rejects_untyped_decimal_division_before_cudf_execution() {
    let dir = unique_temp_dir("nexus-query-engine-native-execution-untyped-decimal-divide-reject");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
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
    let filter = builder.add_filter(source, untyped_divide, Arc::clone(&input_schema));
    let plan = builder.finish_with_host_sink(filter);
    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime));

    let error = match execute_query_plan(&context, &plan) {
        Ok(_) => panic!("native plan validation should reject decimal division without a declared output type"),
        Err(error) => error,
    };
    assert_eq!(error.code(), QueryErrorCode::UnsupportedType);
    assert_eq!(error.operation(), "nexus_query_engine.plan.validate_expr");
    assert_eq!(error.fact_value("left_type"), Some("Decimal128(38, 8)"));
    assert_eq!(error.fact_value("right_type"), Some("Decimal128(38, 8)"));
    assert_eq!(error.fact_value("operator"), Some("/"));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_query_executes_round_scalar_expression() {
    let dir = unique_temp_dir("nexus-query-engine-native-execution-round-scalar");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
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
    let projection = builder.add_projection(
        source,
        projection_outputs(
            vec![
                Expr::round(Expr::literal(LiteralValue::Float64(1.235)), 2, DataType::Float64).alias("rounded_literal"),
                Expr::round(ratio, 1, DataType::Decimal128(38, 1)).alias("rounded_ratio"),
            ],
            output_schema,
        ),
    );
    let plan = builder.finish_with_host_sink(projection);
    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime));

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
    let dir = unique_temp_dir("nexus-query-engine-native-execution-round-decimal-integral-cast");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
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
    let projection = builder.add_projection(source, projection_outputs(vec![rounded_segment], output_schema));
    let plan = builder.finish_with_host_sink(projection);
    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime));

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
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = Arc::new(Schema::new(vec![Field::new("event_date64", DataType::Date64, true)]));

    let cases = [
        (
            "extract_date64",
            vec![Expr::extract(nexus_query_engine::DatePart::Year, Expr::column("event_date64")).alias("event_year")],
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
        let source = LocalParquetSource::new(
            format!("/tmp/does-not-need-to-exist-for-date64-{name}.parquet"),
            Arc::clone(&input_schema),
            vec!["event_date64".to_string()],
        );
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(source));
        let projection = builder.add_projection(source, projection_outputs(projection, output_schema));
        let plan = builder.finish_with_host_sink(projection);
        let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime));

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
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
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
        let source = LocalParquetSource::new(
            format!("/tmp/does-not-need-to-exist-for-substring-{name}.parquet"),
            Arc::clone(&input_schema),
            vec!["orderkey".to_string(), "status".to_string()],
        );
        let output_schema = Arc::new(Schema::new(vec![Field::new("bad_substring", DataType::Utf8, true)]));

        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(source));
        let projection = builder.add_projection(source, projection_outputs(vec![expr], output_schema));
        let plan = builder.finish_with_host_sink(projection);
        let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime));

        let error = match execute_query_plan(&context, &plan) {
            Ok(_) => panic!("{name} should reject before source read"),
            Err(error) => error,
        };
        assert_eq!(error.code(), expected_code, "{name}");
        assert_eq!(error.operation(), "nexus_query_engine.plan.validate_expr", "{name}");
        assert_eq!(error.fact_value("context"), expected_context, "{name}");
        assert!(error.fact_value("path").is_none(), "{name} should fail before source read");
    }
}

#[test]
fn native_query_rejects_unread_source_column_in_nested_scalar_before_cudf_execution() {
    let dir = unique_temp_dir("nexus-query-engine-native-execution-scalar_expression-source-column-reject");
    let path = dir.join("tpch-like.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_scalar_expression_parquet(&runtime, &path);
    let output_schema = Arc::new(Schema::new(vec![Field::new("status_label", DataType::Utf8, true)]));
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["orderkey".to_string()],
    );
    let label = Expr::case(
        vec![nexus_query_engine::CaseWhen::new(
            Expr::like(Expr::column("status"), "O%"),
            Expr::literal(LiteralValue::Utf8("open".to_string())),
        )],
        Expr::literal(LiteralValue::Utf8("other".to_string())),
    )
    .alias("status_label");

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let projection = builder.add_projection(source, projection_outputs(vec![label], output_schema));
    let plan = builder.finish_with_host_sink(projection);
    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime));

    let error = match execute_query_plan(&context, &plan) {
        Ok(_) => panic!("native plan validation should reject unread nested scalar columns before cuDF execution"),
        Err(error) => error,
    };
    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.operation(), "nexus_query_engine.plan.validate_expr");
    assert_eq!(error.fact_value("column"), Some("status"));
    assert!(error.fact_value("path").is_none());

    fs::remove_dir_all(dir).unwrap();
}
