use query_engine::rapids_backend::rapids_analysis_context;

#[test]
fn native_query_executes_gpu_resident_source_with_offset_only_limit() {
    let context = backend_context();
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("label", DataType::Utf8, true),
    ]));
    let input_batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(StringArray::from(vec![Some("one"), None, Some("three")])),
        ],
    )
    .unwrap();

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", Arc::clone(&schema))));
    let limit = builder.add_limit(source, LimitSpec::offset(1), Arc::clone(&schema));
    let plan = builder.finish_with_host_sink(limit);

    let admitted = context.admit_plan(&plan).unwrap();
    let runtime = Arc::new(admitted.runtime().unwrap());
    let mut inputs = GpuResidentInputMap::new();
    inputs.insert(
        "input".to_owned(),
        GpuResultHandle::new(
            &admitted,
            GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &input_batch).unwrap()),
            Arc::clone(&schema),
        )
        .unwrap(),
    );
    let result = execute_query_plan_with_gpu_inputs(&admitted, &plan, &inputs).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.input_batches, 1);
    assert_eq!(metrics.input_rows, 3);
    assert_eq!(metrics.boundary.host_to_device_uploads, 0);
    assert_eq!(metrics.boundary.device_to_host_materializations, 1);
    assert_eq!(metrics.boundary.device_to_host_materialized_rows, 2);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 2);
    assert_eq!(batch.num_columns(), 2);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[2, 3]);
    assert!(string_array(batch, 1).is_null(0));
    assert_eq!(string_array(batch, 1).value(1), "three");
    admitted.finish();
}

#[test]
fn native_gpu_resident_source_preserves_input_order() {
    let context = backend_context();
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", Arc::clone(&schema))));
    let plan = builder.finish_with_host_sink(source);

    let combined_batch =
        RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from(vec![10, 11, 20, 21, 22]))]).unwrap();
    let admitted = context.admit_plan(&plan).unwrap();
    let runtime = Arc::new(admitted.runtime().unwrap());
    let mut inputs = GpuResidentInputMap::new();
    inputs.insert(
        "input".to_owned(),
        GpuResultHandle::new(
            &admitted,
            GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &combined_batch).unwrap()),
            Arc::clone(&schema),
        )
        .unwrap(),
    );
    let result = execute_query_plan_with_gpu_inputs(&admitted, &plan, &inputs).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.input_batches, 1);
    assert_eq!(metrics.input_rows, 5);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 5);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[10, 11, 20, 21, 22]);
    admitted.finish();
}

#[test]
fn native_local_parquet_reader_predicate_coerces_integer_literal_to_float_column_type() {
    let dir = unique_temp_dir("query-engine-native-reader-float-integer-predicate");
    let path = dir.join("input.parquet");
    let context = backend_context();
    let runtime = Arc::new(context.runtime().unwrap());
    let input_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("amount", DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&input_schema),
        vec![Arc::new(Int32Array::from(vec![1, 2, 3])), Arc::new(Float64Array::from(vec![10.5, 20.0, 30.5]))],
    )
    .unwrap();
    DataFrame::from_record_batch(&runtime, &batch).unwrap().write_parquet(&path).unwrap();

    let predicate = Expr::binary(Expr::column("amount"), BinaryOp::Eq, Expr::literal(LiteralValue::Int32(20)));
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["order_id".to_string(), "amount".to_string()],
    )
    .with_reader_predicate(predicate);
    assert!(source.reader_predicate.as_ref().is_some_and(|predicate| source.reader_predicate_is_supported(predicate, rapids_analysis_context())));

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let plan = builder.finish_with_host_sink(source);

    let result = execute_query_plan(&context, &plan).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 1);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[2]);
    assert_eq!(float64_array(batch, 1).values().as_ref(), &[20.0]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_local_parquet_reader_predicate_uses_sql_null_logical_or() {
    let dir = unique_temp_dir("query-engine-native-reader-null-logical-or");
    let path = dir.join("input.parquet");
    let context = backend_context();
    let runtime = Arc::new(context.runtime().unwrap());
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("dmail", DataType::Utf8, true),
        Field::new("email", DataType::Utf8, true),
        Field::new("tv", DataType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5, 6])),
            Arc::new(StringArray::from(vec![Some("Y"), Some("N"), Some("N"), None, Some("N"), None])),
            Arc::new(StringArray::from(vec![None, Some("Y"), None, Some("Y"), Some("N"), None])),
            Arc::new(StringArray::from(vec![Some("N"), None, Some("N"), Some("N"), Some("N"), None])),
        ],
    )
    .unwrap();
    DataFrame::from_record_batch(&runtime, &batch).unwrap().write_parquet(&path).unwrap();

    let y = || Expr::literal(LiteralValue::Utf8("Y".to_string()));
    let dmail_is_y = Expr::binary(Expr::column("dmail"), BinaryOp::Eq, y());
    let email_is_y = Expr::binary(Expr::column("email"), BinaryOp::Eq, y());
    let tv_is_y = Expr::binary(Expr::column("tv"), BinaryOp::Eq, y());
    let predicate = Expr::binary(Expr::binary(dmail_is_y, BinaryOp::Or, email_is_y), BinaryOp::Or, tv_is_y);
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&schema),
        vec!["key".to_string(), "dmail".to_string(), "email".to_string(), "tv".to_string()],
    )
    .with_column_roles(vec!["key".to_string()], vec!["dmail".to_string(), "email".to_string(), "tv".to_string()])
    .with_reader_predicate(predicate);

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let plan = builder.finish_with_host_sink(source);

    let result = execute_query_plan(&context, &plan).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 3);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[1, 2, 4]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_filter_predicate_uses_where_semantics_for_nullable_and_or_in_lists() {
    let context = backend_context();
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("a", DataType::Boolean, true),
        Field::new("b", DataType::Boolean, true),
        Field::new("label", DataType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5, 6])),
            Arc::new(BooleanArray::from(vec![Some(true), Some(true), Some(false), None, None, None])),
            Arc::new(BooleanArray::from(vec![Some(true), None, None, Some(true), Some(false), None])),
            Arc::new(StringArray::from(vec![Some("A"), Some("C"), None, Some("B"), Some("D"), Some("A")])),
        ],
    )
    .unwrap();

    assert_eq!(
        execute_gpu_resident_filter_ids(
            &context,
            Arc::clone(&schema),
            &batch,
            Expr::binary(Expr::column("a"), BinaryOp::And, Expr::column("b"))
        ),
        vec![1]
    );
    assert_eq!(
        execute_gpu_resident_filter_ids(
            &context,
            Arc::clone(&schema),
            &batch,
            Expr::binary(Expr::column("a"), BinaryOp::Or, Expr::column("b"))
        ),
        vec![1, 2, 4]
    );
    assert_eq!(
        execute_gpu_resident_filter_ids(
            &context,
            Arc::clone(&schema),
            &batch,
            Expr::in_list(
                Expr::column("label"),
                vec![
                    Expr::literal(LiteralValue::Utf8("A".to_string())),
                    Expr::literal(LiteralValue::Utf8("B".to_string())),
                ],
            ),
        ),
        vec![1, 4, 6]
    );
    assert_eq!(
        execute_gpu_resident_filter_ids(
            &context,
            Arc::clone(&schema),
            &batch,
            Expr::not_in_list(
                Expr::column("label"),
                vec![
                    Expr::literal(LiteralValue::Utf8("A".to_string())),
                    Expr::literal(LiteralValue::Utf8("B".to_string())),
                ],
            ),
        ),
        vec![2, 5]
    );
}

#[test]
fn native_filter_predicate_executes_large_numeric_literal_in_list() {
    let context = backend_context();
    let schema =
        Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false), Field::new("key", DataType::Int64, true)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5, 6])),
            Arc::new(Int64Array::from(vec![Some(2), Some(3), Some(128), None, Some(250), Some(260)])),
        ],
    )
    .unwrap();
    let list = (0..129).map(|value| Expr::literal(LiteralValue::Int64(value * 2))).collect();

    assert_eq!(
        execute_gpu_resident_filter_ids(
            &context,
            Arc::clone(&schema),
            &batch,
            Expr::in_list(Expr::column("key"), list)
        ),
        vec![1, 3, 5]
    );
}
