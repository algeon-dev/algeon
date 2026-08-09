#[test]
fn native_query_executes_group_only_aggregate_as_distinct_keys() {
    let dir = unique_temp_dir("nexus-query-engine-native-group-only-aggregate");
    let path = dir.join("input.parquet");
    let context = backend_context();
    let runtime = Arc::new(context.runtime().unwrap());
    let input_schema = write_input_parquet(&runtime, &path);
    let output_schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, false)]));

    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["key".to_string()],
    );
    let aggregate = AggregateSpec::new(vec![Expr::column("key")], vec![]);
    let sort = SortSpec::new(vec![SortExpr::new(Expr::column("key"), false, false)]);

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let aggregate = builder.add_aggregate(source, aggregate, Arc::clone(&output_schema));
    let sort = builder.add_sort(aggregate, sort, Arc::clone(&output_schema));
    let plan = builder.finish_with_host_sink(sort);

    let result = execute_query_plan(&context, &plan).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 3);
    assert_eq!(batch.num_columns(), 1);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[1, 2, 3]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_executes_group_only_aggregate_with_budgeted_distinct() {
    let dir = unique_temp_dir("nexus-query-engine-bounded-group-only-aggregate");
    let path = dir.join("input.parquet");
    let context = backend_context();
    let input_schema = write_final_merge_retry_input_parquet(&path, 4);
    let output_schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, false)]));

    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["key".to_string()],
    );
    let aggregate = AggregateSpec::new(vec![Expr::column("key")], vec![]);
    let sort = SortSpec::new(vec![SortExpr::new(Expr::column("key"), false, false)]);

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let aggregate = builder.add_aggregate(source, aggregate, Arc::clone(&output_schema));
    let sort = builder.add_sort(aggregate, sort, Arc::clone(&output_schema));
    let plan = builder.finish_with_host_sink(sort);

    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.source_chunks_committed, 3);
    assert_eq!(metrics.final_merge_attempts, 1);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 4);
    assert_eq!(batch.num_columns(), 1);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[0, 1, 2, 3]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_query_executes_group_only_aggregate_when_output_key_name_already_exists() {
    let context = backend_context();
    let schema = Arc::new(Schema::new(vec![
        Field::new("i_item_id", DataType::Utf8, false),
        Field::new("__nexus_native_group_expr_0_i_item_id", DataType::Utf8, false),
        Field::new("i_item_desc", DataType::Utf8, false),
        Field::new("i_current_price", DataType::Int32, false),
    ]));
    let input_batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(vec!["carry_a", "carry_b", "carry_c", "carry_d"])),
            Arc::new(StringArray::from(vec!["item_1", "item_1", "item_2", "item_2"])),
            Arc::new(StringArray::from(vec!["desc_1", "desc_1", "desc_2", "desc_2"])),
            Arc::new(Int32Array::from(vec![10, 10, 20, 20])),
        ],
    )
    .unwrap();
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("i_item_id", DataType::Utf8, false),
        Field::new("i_item_desc", DataType::Utf8, false),
        Field::new("i_current_price", DataType::Int32, false),
    ]));
    let aggregate = AggregateSpec::new(
        vec![
            Expr::column("__nexus_native_group_expr_0_i_item_id").alias("i_item_id"),
            Expr::column("i_item_desc"),
            Expr::column("i_current_price"),
        ],
        vec![],
    );
    let sort = SortSpec::new(vec![SortExpr::new(Expr::column("i_item_id"), false, false)]);

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", Arc::clone(&schema))));
    let aggregate = builder.add_aggregate(source, aggregate, Arc::clone(&output_schema));
    let sort = builder.add_sort(aggregate, sort, Arc::clone(&output_schema));
    let plan = builder.finish_with_host_sink(sort);

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
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 2);
    assert_eq!(batch.num_columns(), 3);
    assert_eq!(string_array(batch, 0).iter().collect::<Vec<_>>(), vec![Some("item_1"), Some("item_2")]);
    assert_eq!(string_array(batch, 1).iter().collect::<Vec<_>>(), vec![Some("desc_1"), Some("desc_2")]);
    assert_eq!(int32_array(batch, 2).values().as_ref(), &[10, 20]);
    admitted.finish();
}
