use std::sync::Arc;

use super::*;

#[test]
fn native_boolean_and_or_use_sql_three_valued_logic() {
    let context = backend_context();
    let input_schema =
        Arc::new(Schema::new(vec![Field::new("a", DataType::Boolean, true), Field::new("b", DataType::Boolean, true)]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("a_or_b", DataType::Boolean, true),
        Field::new("a_and_b", DataType::Boolean, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&input_schema),
        vec![
            Arc::new(BooleanArray::from(vec![Some(false), Some(false), Some(true), Some(true), None, None, None])),
            Arc::new(BooleanArray::from(vec![
                Some(false),
                Some(true),
                Some(false),
                Some(true),
                Some(false),
                Some(true),
                None,
            ])),
        ],
    )
    .unwrap();
    let mut builder = QueryPlanBuilder::new();
    let source =
        builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("bool_input", Arc::clone(&input_schema))));
    let projection = builder.add_projection(
        source,
        projection_outputs(
            vec![
                Expr::binary(Expr::column("a"), BinaryOp::Or, Expr::column("b")).alias("a_or_b"),
                Expr::binary(Expr::column("a"), BinaryOp::And, Expr::column("b")).alias("a_and_b"),
            ],
            Arc::clone(&output_schema),
        ),
    );
    let plan = builder.finish_with_host_sink(projection);
    let admitted = context.admit_plan(&plan).unwrap();
    let runtime = Arc::new(admitted.runtime().unwrap());
    let frame = DataFrame::from_record_batch(&runtime, &batch).unwrap();

    let mut gpu_inputs = GpuResidentInputMap::new();
    gpu_inputs.insert(
        "bool_input".to_string(),
        GpuResultHandle::new(&admitted, GpuDataFrame::new(frame), Arc::clone(&input_schema)).unwrap(),
    );

    let result = execute_query_plan_with_gpu_inputs(&admitted, &plan, &gpu_inputs).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };

    assert_boolean_values(
        boolean_array(batch, 0),
        &[Some(false), Some(true), Some(true), Some(true), None, Some(true), None],
    );
    assert_boolean_values(
        boolean_array(batch, 1),
        &[Some(false), Some(false), Some(false), Some(true), Some(false), None, None],
    );
    admitted.finish();
}

fn assert_boolean_values(array: &BooleanArray, expected: &[Option<bool>]) {
    assert_eq!(array.len(), expected.len());
    for (index, expected) in expected.iter().enumerate() {
        match expected {
            Some(expected) => {
                assert!(!array.is_null(index), "row {index} should be non-null");
                assert_eq!(array.value(index), *expected, "row {index} value mismatch");
            },
            None => assert!(array.is_null(index), "row {index} should be null"),
        }
    }
}

#[test]
fn native_query_executes_filter_projection_over_tpch_like_types() {
    let dir = unique_temp_dir("query-engine-native-execution-tpch-like-types");
    let path = dir.join("tpch-like.parquet");
    let context = backend_context();
    let runtime = Arc::new(context.runtime().unwrap());
    let input_schema = write_tpch_like_types_parquet(&runtime, &path);
    let projected_schema = Arc::new(Schema::new(vec![
        Field::new("orderkey", DataType::Int64, false),
        Field::new("amount", DataType::Decimal128(15, 2), false),
        Field::new("shipdate", DataType::Date32, false),
        Field::new("shipdate_plus_month", DataType::Date32, false),
        Field::new("event_ts", DataType::Timestamp(TimeUnit::Nanosecond, None), false),
        Field::new("event_ts_plus", DataType::Timestamp(TimeUnit::Nanosecond, None), false),
        Field::new("event_ts_plus_interval", DataType::Timestamp(TimeUnit::Nanosecond, None), false),
        Field::new("status", DataType::Utf8, false),
        Field::new("large_literal", DataType::Utf8, false),
        Field::new("null_amount", DataType::Decimal128(15, 2), true),
        Field::new("null_shipdate", DataType::Date32, true),
        Field::new("null_event_ts", DataType::Timestamp(TimeUnit::Nanosecond, None), true),
        Field::new("null_status", DataType::Utf8, true),
    ]));
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec![
            "orderkey".to_string(),
            "amount".to_string(),
            "shipdate".to_string(),
            "event_ts".to_string(),
            "status".to_string(),
        ],
    );
    let filter = Expr::binary(
        Expr::binary(
            Expr::binary(
                Expr::binary(
                    Expr::column("amount"),
                    BinaryOp::Gt,
                    Expr::literal(LiteralValue::Decimal128 { raw: 20_000, precision: 15, scale: 2 }),
                ),
                BinaryOp::And,
                Expr::binary(Expr::column("shipdate"), BinaryOp::GtEq, Expr::literal(LiteralValue::Date32(18_322))),
            ),
            BinaryOp::And,
            Expr::binary(
                Expr::column("event_ts"),
                BinaryOp::Lt,
                Expr::literal(LiteralValue::Timestamp { raw: 400, unit: TimeUnit::Nanosecond }),
            ),
        ),
        BinaryOp::And,
        Expr::binary(Expr::column("status"), BinaryOp::Eq, Expr::literal(LiteralValue::Utf8("O".to_string()))),
    );
    let projection = vec![
        Expr::column("orderkey"),
        Expr::column("amount"),
        Expr::column("shipdate"),
        Expr::binary(
            Expr::column("shipdate"),
            BinaryOp::Add,
            Expr::literal(LiteralValue::IntervalMonthDayNano { months: 1, days: 0, nanoseconds: 0 }),
        )
        .alias("shipdate_plus_month"),
        Expr::column("event_ts"),
        Expr::binary(
            Expr::column("event_ts"),
            BinaryOp::Add,
            Expr::literal(LiteralValue::Duration { raw: 10, unit: TimeUnit::Nanosecond }),
        )
        .alias("event_ts_plus"),
        Expr::binary(
            Expr::column("event_ts"),
            BinaryOp::Add,
            Expr::literal(LiteralValue::IntervalMonthDayNano { months: 0, days: 1, nanoseconds: 5 }),
        )
        .alias("event_ts_plus_interval"),
        Expr::column("status"),
        Expr::literal(LiteralValue::LargeUtf8("wide".to_string())).alias("large_literal"),
        Expr::literal(LiteralValue::Null(DataType::Decimal128(15, 2))).alias("null_amount"),
        Expr::literal(LiteralValue::Null(DataType::Date32)).alias("null_shipdate"),
        Expr::literal(LiteralValue::Null(DataType::Timestamp(TimeUnit::Nanosecond, None))).alias("null_event_ts"),
        Expr::literal(LiteralValue::Null(DataType::Utf8)).alias("null_status"),
    ];

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let filter = builder.add_filter(source, filter, Arc::clone(&input_schema));
    let projection = builder.add_projection(filter, projection_outputs(projection, projected_schema));
    let plan = builder.finish_with_host_sink(projection);

    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.parquet_files, 1);
    assert_eq!(metrics.read_columns, 5);
    assert_eq!(metrics.boundary.host_to_device_uploads, 0);
    assert_eq!(metrics.boundary.device_to_host_materializations, 1);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 1);
    assert_eq!(int64_array(batch, 0).value(0), 3);
    let amount = decimal128_array(batch, 1);
    assert_eq!(amount.value(0), 22_222);
    assert!(matches!(amount.data_type(), DataType::Decimal128(_, 2)));
    assert_eq!(date32_array(batch, 2).value(0), 18_322);
    assert_eq!(date32_array(batch, 3).value(0), 18_353);
    assert_eq!(timestamp_ns_array(batch, 4).value(0), 300);
    assert_eq!(timestamp_ns_array(batch, 5).value(0), 310);
    assert_eq!(timestamp_ns_array(batch, 6).value(0), 86_400_000_000_305);
    assert_eq!(string_array(batch, 7).value(0), "O");
    assert_eq!(string_array(batch, 8).value(0), "wide");
    for index in 9..13 {
        assert!(batch.column(index).is_null(0), "projection column {index} should be a typed null");
    }

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_projection_coerces_decimal_output_to_declared_scale() {
    let context = backend_context();
    let input_schema = Arc::new(Schema::new(vec![Field::new("amount", DataType::Decimal128(20, 0), false)]));
    let output_schema = Arc::new(Schema::new(vec![Field::new("mkt_share", DataType::Decimal128(20, -1), false)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&input_schema),
        vec![Arc::new(Decimal128Array::from_iter_values([120, 340]).with_precision_and_scale(20, 0).unwrap())],
    )
    .unwrap();
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::GpuResident(
        GpuResidentSource::new("decimal_input", Arc::clone(&input_schema))
            .with_row_count_upper_bound(batch.num_rows() as u64),
    ));
    let projection = builder.add_projection(
        source,
        projection_outputs(vec![Expr::column("amount").alias("mkt_share")], Arc::clone(&output_schema)),
    );
    let plan = builder.finish_gpu_resident(projection);
    let admitted = context.admit_plan(&plan).unwrap();
    let runtime = Arc::new(admitted.runtime().unwrap());
    let frame = DataFrame::from_record_batch(&runtime, &batch).unwrap();

    let mut gpu_inputs = GpuResidentInputMap::new();
    gpu_inputs.insert(
        "decimal_input".to_string(),
        GpuResultHandle::new(&admitted, GpuDataFrame::new(frame), Arc::clone(&input_schema)).unwrap(),
    );

    let result = execute_query_plan_with_gpu_inputs(&admitted, &plan, &gpu_inputs).unwrap();
    let QueryOutput::GpuResident(handle) = result.output() else {
        panic!("GPU-resident sink should keep projection output on device");
    };
    assert_eq!(handle.schema().as_ref(), output_schema.as_ref());

    let mut materialization_metrics = QueryMetrics::default();
    let batch = materialize_gpu_result_to_host(handle.clone(), &mut materialization_metrics).unwrap();
    assert_eq!(batch.schema().as_ref(), output_schema.as_ref());
    let mkt_share = decimal128_array(&batch, 0);
    assert_eq!(mkt_share.data_type(), &DataType::Decimal128(20, -1));
    assert_eq!(mkt_share.values().as_ref(), &[12, 34]);
    admitted.finish();
}

#[test]
fn bounded_native_projection_coerces_decimal_output_to_declared_scale() {
    let dir = unique_temp_dir("query-engine-bounded-projection-decimal-schema-coerce");
    let path = dir.join("input.parquet");
    let context = backend_context();
    let input_schema = write_decimal_scale0_grouped_parquet(&path);
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("sum_amount", DataType::Decimal128(20, 0), true),
    ]));
    let projection_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("mkt_share", DataType::Decimal128(20, -1), true),
    ]));
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["key".to_string(), "amount".to_string()],
    );
    let aggregate = AggregateSpec::new(
        vec![Expr::column("key")],
        vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("amount"), "sum_amount")],
    );

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let aggregate = builder.add_aggregate(source, aggregate, aggregate_schema);
    let projection = builder.add_projection(
        aggregate,
        projection_outputs(
            vec![Expr::column("key"), Expr::column("sum_amount").alias("mkt_share")],
            Arc::clone(&projection_schema),
        ),
    );
    let plan = builder.finish_gpu_resident(projection);

    let result = execute_query_plan(&context, &plan).unwrap();
    assert!(result.metrics().budgeted_execution_events.iter().any(|event| {
        event.operation == "query_engine.projection.schema_coerce" && event.boundary == "post_aggregate_projection"
    }));
    let QueryOutput::GpuResident(handle) = result.output() else {
        panic!("GPU-resident sink should keep bounded projection output on device");
    };
    assert_eq!(handle.schema().as_ref(), projection_schema.as_ref());

    let mut materialization_metrics = QueryMetrics::default();
    let batch = materialize_gpu_result_to_host(handle.clone(), &mut materialization_metrics).unwrap();
    assert_eq!(batch.schema().as_ref(), projection_schema.as_ref());
    let mkt_share = decimal128_array(&batch, 1);
    assert_eq!(mkt_share.data_type(), &DataType::Decimal128(20, -1));
    let mut values = mkt_share.values().iter().copied().collect::<Vec<_>>();
    values.sort_unstable();
    assert_eq!(values, vec![46, 50]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_projection_decimal_division_uses_declared_binary_output_type() {
    let context = backend_context();
    let input_schema = Arc::new(Schema::new(vec![
        Field::new("numerator", DataType::Decimal128(38, 8), false),
        Field::new("denominator", DataType::Decimal128(38, 8), false),
    ]));
    let output_schema = Arc::new(Schema::new(vec![Field::new("ratio", DataType::Decimal128(38, 8), false)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&input_schema),
        vec![
            Arc::new(
                Decimal128Array::from_iter_values([100_000_000, 300_000_000]).with_precision_and_scale(38, 8).unwrap(),
            ),
            Arc::new(
                Decimal128Array::from_iter_values([400_000_000, 600_000_000]).with_precision_and_scale(38, 8).unwrap(),
            ),
        ],
    )
    .unwrap();
    let projection_expr = Expr::binary_with_output_type(
        Expr::column("numerator"),
        BinaryOp::Divide,
        Expr::column("denominator"),
        DataType::Decimal128(38, 8),
    )
    .alias("ratio");
    let mut builder = QueryPlanBuilder::new();
    let source =
        builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("decimal_input", Arc::clone(&input_schema))));
    let projection =
        builder.add_projection(source, projection_outputs(vec![projection_expr], Arc::clone(&output_schema)));
    let plan = builder.finish_with_host_sink(projection);
    let admitted = context.admit_plan(&plan).unwrap();
    let runtime = Arc::new(admitted.runtime().unwrap());
    let frame = DataFrame::from_record_batch(&runtime, &batch).unwrap();

    let mut gpu_inputs = GpuResidentInputMap::new();
    gpu_inputs.insert(
        "decimal_input".to_string(),
        GpuResultHandle::new(&admitted, GpuDataFrame::new(frame), Arc::clone(&input_schema)).unwrap(),
    );

    let result = execute_query_plan_with_gpu_inputs(&admitted, &plan, &gpu_inputs).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.schema().as_ref(), output_schema.as_ref());
    let ratio = decimal128_array(batch, 0);
    assert_eq!(ratio.data_type(), &DataType::Decimal128(38, 8));
    assert_eq!(ratio.values().as_ref(), &[25_000_000, 50_000_000]);
    admitted.finish();
}

#[test]
fn bounded_native_projection_decimal_division_uses_declared_binary_output_type() {
    let dir = unique_temp_dir("query-engine-bounded-projection-decimal-division");
    let path = dir.join("input.parquet");
    let context = backend_context();
    let input_schema = write_decimal_ratio_parquet(&path);
    let output_schema = Arc::new(Schema::new(vec![Field::new("ratio", DataType::Decimal128(38, 8), false)]));
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["numerator".to_string(), "denominator".to_string()],
    );

    let projection_expr = Expr::binary_with_output_type(
        Expr::column("numerator"),
        BinaryOp::Divide,
        Expr::column("denominator"),
        DataType::Decimal128(38, 8),
    )
    .alias("ratio");
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let projection =
        builder.add_projection(source, projection_outputs(vec![projection_expr], Arc::clone(&output_schema)));
    let plan = builder.finish_with_host_sink(projection);

    let result = execute_query_plan(&context, &plan).unwrap();
    assert!(
        result
            .metrics()
            .budgeted_execution_events
            .iter()
            .any(|event| event.operation == "query_engine.projection.execute")
    );
    assert!(
        !result
            .metrics()
            .budgeted_execution_events
            .iter()
            .any(|event| event.operation == "query_engine.projection.schema_coerce")
    );
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.schema().as_ref(), output_schema.as_ref());
    let ratio = decimal128_array(batch, 0);
    assert_eq!(ratio.data_type(), &DataType::Decimal128(38, 8));
    assert_eq!(ratio.values().as_ref(), &[25_000_000, 50_000_000]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_executes_decimal_sum_ratio_at_output_scale() {
    let dir = unique_temp_dir("query-engine-bounded-market-share-decimal-ratio");
    let path = dir.join("input.parquet");
    let context = backend_context();
    let input_schema = write_market_share_decimal_ratio_parquet(&path);
    let projected_schema = Arc::new(Schema::new(vec![
        Field::new("o_year", DataType::Int32, false),
        Field::new("target_volume", DataType::Decimal128(38, 8), true),
        Field::new("volume", DataType::Decimal128(38, 8), false),
    ]));
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("o_year", DataType::Int32, false),
        Field::new("sum_target_volume", DataType::Decimal128(38, 8), true),
        Field::new("sum_volume", DataType::Decimal128(38, 8), true),
    ]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("o_year", DataType::Int32, false),
        Field::new("mkt_share", DataType::Decimal128(38, 8), true),
    ]));
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["o_year".to_string(), "is_target".to_string(), "volume".to_string()],
    );
    let target_volume = Expr::case(
        vec![query_engine::expr::CaseWhen::new(Expr::column("is_target"), Expr::column("volume"))],
        Expr::literal(LiteralValue::Decimal128 { raw: 0, precision: 38, scale: 8 }),
    )
    .alias("target_volume");
    let aggregate = AggregateSpec::new(
        vec![Expr::column("o_year")],
        vec![
            AggregateExpr::new(AggregateFunction::Sum, Expr::column("target_volume"), "sum_target_volume"),
            AggregateExpr::new(AggregateFunction::Sum, Expr::column("volume"), "sum_volume"),
        ],
    );
    let mkt_share = Expr::binary(Expr::column("sum_target_volume"), BinaryOp::Divide, Expr::column("sum_volume"))
        .alias("mkt_share");

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let projection = builder.add_projection(
        source,
        projection_outputs(vec![Expr::column("o_year"), target_volume, Expr::column("volume")], projected_schema),
    );
    let aggregate = builder.add_aggregate(projection, aggregate, aggregate_schema);
    let projection =
        builder.add_projection(aggregate, projection_outputs(vec![Expr::column("o_year"), mkt_share], output_schema));
    let plan = builder.finish_with_host_sink(projection);

    let result = execute_query_plan(&context, &plan).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    let years = int32_array(batch, 0);
    let shares = decimal128_array(batch, 1);
    let mut rows = (0..batch.num_rows()).map(|row| (years.value(row), shares.value(row))).collect::<Vec<_>>();
    rows.sort_unstable();
    assert_eq!(shares.data_type(), &DataType::Decimal128(38, 8));
    assert_eq!(rows, vec![(2020, 25_000_000), (2021, 50_000_000)]);

    fs::remove_dir_all(dir).unwrap();
}
