#[test]
fn native_query_executes_group_only_aggregate_as_distinct_keys() {
    let dir = unique_temp_dir("query-engine-native-group-only-aggregate");
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
    let dir = unique_temp_dir("query-engine-bounded-group-only-aggregate");
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
        Field::new("__algeon_native_group_expr_0_i_item_id", DataType::Utf8, false),
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
            Expr::column("__algeon_native_group_expr_0_i_item_id").alias("i_item_id"),
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

#[test]
fn native_group_only_distinct_preserves_schema_over_empty_join_output() {
    let dir = unique_temp_dir("query-engine-empty-join-group-only-distinct");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    let context = backend_context();
    let runtime = Arc::new(context.runtime().unwrap());
    let left_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("semantic", DataType::Utf8, false),
    ]));
    let right_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("vertexId", DataType::Utf8, false),
        Field::new("vertexPriority", DataType::Int64, false),
    ]));
    let left_batch = RecordBatch::try_new(
        Arc::clone(&left_schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(StringArray::from(vec!["depends", "owns", "calls"])),
        ],
    )
    .unwrap();
    let right_batch = RecordBatch::try_new(
        Arc::clone(&right_schema),
        vec![
            Arc::new(Int32Array::from(vec![99])),
            Arc::new(StringArray::from(vec!["vertex-99"])),
            Arc::new(Int64Array::from(vec![7])),
        ],
    )
    .unwrap();
    DataFrame::from_record_batch(&runtime, &left_batch).unwrap().write_parquet(&left_path).unwrap();
    DataFrame::from_record_batch(&runtime, &right_batch).unwrap().write_parquet(&right_path).unwrap();

    let join_schema = Arc::new(Schema::new(vec![
        Field::new("semantic", DataType::Utf8, false),
        Field::new("vertexId", DataType::Utf8, false),
        Field::new("vertexPriority", DataType::Int64, false),
    ]));
    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&left_path).with_file_size_bytes(fs::metadata(&left_path).unwrap().len())],
        Arc::clone(&left_schema),
        vec!["key".to_owned(), "semantic".to_owned()],
    )));
    let right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&right_path).with_file_size_bytes(fs::metadata(&right_path).unwrap().len())],
        Arc::clone(&right_schema),
        vec!["key".to_owned(), "vertexId".to_owned(), "vertexPriority".to_owned()],
    )));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                query_engine::expr::ColumnRef::input(0, "key", "key"),
                query_engine::expr::ColumnRef::input(1, "key", "key"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(
                query_engine::expr::ColumnRef::input(0, "semantic", "semantic"),
                "semantic",
            ),
            JoinOutputColumn::right(query_engine::expr::ColumnRef::input(1, "vertexId", "vertexId"), "vertexId"),
            JoinOutputColumn::right(
                query_engine::expr::ColumnRef::input(1, "vertexPriority", "vertexPriority"),
                "vertexPriority",
            ),
        ])
        .with_build_side(JoinSide::Right),
        Arc::clone(&join_schema),
    );
    let aggregate = builder.add_aggregate(
        join,
        AggregateSpec::new(
            vec![Expr::column("semantic"), Expr::column("vertexId"), Expr::column("vertexPriority")],
            vec![],
        ),
        Arc::clone(&join_schema),
    );
    let limit = builder.add_limit(aggregate, LimitSpec::fetch(100), Arc::clone(&join_schema));
    let plan = builder.finish_with_host_sink(limit);

    let result = execute_query_plan(&context, &plan).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.schema(), join_schema);
    assert_eq!(batch.num_rows(), 0);

    let matching_right_batch = RecordBatch::try_new(
        Arc::clone(&right_schema),
        vec![
            Arc::new(Int32Array::from(vec![2])),
            Arc::new(StringArray::from(vec!["vertex-99"])),
            Arc::new(Int64Array::from(vec![7])),
        ],
    )
    .unwrap();
    fs::remove_file(&right_path).unwrap();
    DataFrame::from_record_batch(&runtime, &matching_right_batch)
        .unwrap()
        .write_parquet(&right_path)
        .unwrap();
    let nonempty = execute_query_plan(&context, &plan).unwrap();
    let QueryOutput::HostArrow(nonempty_batch) = nonempty.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(nonempty_batch.schema(), join_schema);
    assert_eq!(nonempty_batch.num_rows(), 1);

    fs::remove_dir_all(dir).unwrap();
}
