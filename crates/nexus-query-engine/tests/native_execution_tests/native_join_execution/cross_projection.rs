use super::*;

#[test]
fn native_query_executes_cross_join_with_explicit_output_projection() {
    let dir = unique_temp_dir("nexus-query-engine-native-cross-join");
    let left_path = dir.join("lineitem.parquet");
    let right_path = dir.join("orders.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let left_schema = write_join_left_parquet(&runtime, &left_path);
    let right_schema = write_join_right_parquet(&runtime, &right_path);
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("qty", DataType::Int32, false),
        Field::new("category", DataType::Utf8, false),
    ]));

    let left_source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&left_path).with_file_size_bytes(fs::metadata(&left_path).unwrap().len())],
        Arc::clone(&left_schema),
        vec!["qty".to_string()],
    );
    let right_source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&right_path).with_file_size_bytes(fs::metadata(&right_path).unwrap().len())],
        Arc::clone(&right_schema),
        vec!["category".to_string()],
    );

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(left_source));
    let right = builder.add_source(SourceNode::LocalParquet(right_source));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(JoinType::Cross, vec![]).with_output_projection(vec![
            JoinOutputColumn::left(nexus_query_engine::ColumnRef::input(0, "qty", "qty"), "qty"),
            JoinOutputColumn::right(nexus_query_engine::ColumnRef::input(1, "category", "category"), "category"),
        ]),
        Arc::clone(&output_schema),
    );
    let sort = builder.add_sort(
        join,
        SortSpec::new(vec![
            SortExpr::new(Expr::column("qty"), false, false),
            SortExpr::new(Expr::column("category"), false, false),
        ]),
        Arc::clone(&output_schema),
    );
    let plan = builder.finish_with_host_sink(sort);

    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime)).unwrap();
    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    let join_metrics =
        metrics.operators.iter().find_map(|operator| operator.join.as_ref()).expect("join metrics should be recorded");
    assert_eq!(join_metrics.join_type, "cross");
    assert_eq!(join_metrics.key_count, 0);
    assert_eq!(join_metrics.output_rows, 12);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 12);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[10, 10, 10, 20, 20, 20, 25, 25, 25, 30, 30, 30]);
    assert_eq!(
        string_array(batch, 1).iter().collect::<Vec<_>>(),
        vec![
            Some("four"),
            Some("three"),
            Some("two"),
            Some("four"),
            Some("three"),
            Some("two"),
            Some("four"),
            Some("three"),
            Some("two"),
            Some("four"),
            Some("three"),
            Some("two"),
        ]
    );

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_query_preserves_cross_join_right_then_left_projection_order() {
    let dir = unique_temp_dir("nexus-query-engine-native-cross-join-right-left-order");
    let left_path = dir.join("lineitem.parquet");
    let right_path = dir.join("orders.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let left_schema = write_join_left_parquet(&runtime, &left_path);
    let right_schema = write_join_right_parquet(&runtime, &right_path);
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("category", DataType::Utf8, false),
        Field::new("qty", DataType::Int32, false),
    ]));

    let left_source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&left_path).with_file_size_bytes(fs::metadata(&left_path).unwrap().len())],
        Arc::clone(&left_schema),
        vec!["qty".to_string()],
    );
    let right_source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&right_path).with_file_size_bytes(fs::metadata(&right_path).unwrap().len())],
        Arc::clone(&right_schema),
        vec!["category".to_string()],
    );

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(left_source));
    let right = builder.add_source(SourceNode::LocalParquet(right_source));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(JoinType::Cross, vec![]).with_output_projection(vec![
            JoinOutputColumn::right(nexus_query_engine::ColumnRef::input(1, "category", "category"), "category"),
            JoinOutputColumn::left(nexus_query_engine::ColumnRef::input(0, "qty", "qty"), "qty"),
        ]),
        Arc::clone(&output_schema),
    );
    let sort = builder.add_sort(
        join,
        SortSpec::new(vec![
            SortExpr::new(Expr::column("category"), false, false),
            SortExpr::new(Expr::column("qty"), false, false),
        ]),
        Arc::clone(&output_schema),
    );
    let plan = builder.finish_with_host_sink(sort);

    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime)).unwrap();
    let result = execute_query_plan(&context, &plan).unwrap();
    let join_metrics =
        result.metrics().operators.iter().find_map(|operator| operator.join.as_ref()).expect("join metrics");
    assert_eq!(join_metrics.join_type, "cross");
    assert_eq!(join_metrics.output_rows, 12);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.schema().field(0).name(), "category");
    assert_eq!(batch.schema().field(1).name(), "qty");
    let rows = (0..batch.num_rows())
        .map(|row| (string_array(batch, 0).value(row).to_string(), int32_array(batch, 1).value(row)))
        .collect::<Vec<_>>();
    assert_eq!(
        rows,
        vec![
            ("four".to_string(), 10),
            ("four".to_string(), 20),
            ("four".to_string(), 25),
            ("four".to_string(), 30),
            ("three".to_string(), 10),
            ("three".to_string(), 20),
            ("three".to_string(), 25),
            ("three".to_string(), 30),
            ("two".to_string(), 10),
            ("two".to_string(), 20),
            ("two".to_string(), 25),
            ("two".to_string(), 30),
        ]
    );

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_query_preserves_cross_join_interleaved_projection_order() {
    let dir = unique_temp_dir("nexus-query-engine-native-cross-join-interleaved-order");
    let left_path = dir.join("lineitem.parquet");
    let right_path = dir.join("orders.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let left_schema = write_join_left_parquet(&runtime, &left_path);
    let right_schema = write_join_right_parquet(&runtime, &right_path);
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("right_order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
        Field::new("category", DataType::Utf8, false),
        Field::new("left_order_id", DataType::Int32, false),
    ]));

    let left_source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&left_path).with_file_size_bytes(fs::metadata(&left_path).unwrap().len())],
        Arc::clone(&left_schema),
        vec!["order_id".to_string(), "qty".to_string()],
    );
    let right_source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&right_path).with_file_size_bytes(fs::metadata(&right_path).unwrap().len())],
        Arc::clone(&right_schema),
        vec!["order_id".to_string(), "category".to_string()],
    );

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(left_source));
    let right = builder.add_source(SourceNode::LocalParquet(right_source));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(JoinType::Cross, vec![]).with_output_projection(vec![
            JoinOutputColumn::right(
                nexus_query_engine::ColumnRef::input(1, "order_id", "right_order_id"),
                "right_order_id",
            ),
            JoinOutputColumn::left(nexus_query_engine::ColumnRef::input(0, "qty", "qty"), "qty"),
            JoinOutputColumn::right(nexus_query_engine::ColumnRef::input(1, "category", "category"), "category"),
            JoinOutputColumn::left(
                nexus_query_engine::ColumnRef::input(0, "order_id", "left_order_id"),
                "left_order_id",
            ),
        ]),
        Arc::clone(&output_schema),
    );
    let sort = builder.add_sort(
        join,
        SortSpec::new(vec![
            SortExpr::new(Expr::column("right_order_id"), false, false),
            SortExpr::new(Expr::column("qty"), false, false),
            SortExpr::new(Expr::column("category"), false, false),
            SortExpr::new(Expr::column("left_order_id"), false, false),
        ]),
        Arc::clone(&output_schema),
    );
    let plan = builder.finish_with_host_sink(sort);

    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime)).unwrap();
    let result = execute_query_plan(&context, &plan).unwrap();
    let join_metrics =
        result.metrics().operators.iter().find_map(|operator| operator.join.as_ref()).expect("join metrics");
    assert_eq!(join_metrics.join_type, "cross");
    assert_eq!(join_metrics.output_rows, 12);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.schema().field(0).name(), "right_order_id");
    assert_eq!(batch.schema().field(1).name(), "qty");
    assert_eq!(batch.schema().field(2).name(), "category");
    assert_eq!(batch.schema().field(3).name(), "left_order_id");
    let rows = (0..batch.num_rows())
        .map(|row| {
            (
                int32_array(batch, 0).value(row),
                int32_array(batch, 1).value(row),
                string_array(batch, 2).value(row).to_string(),
                int32_array(batch, 3).value(row),
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(
        rows,
        vec![
            (2, 10, "two".to_string(), 1),
            (2, 20, "two".to_string(), 2),
            (2, 25, "two".to_string(), 2),
            (2, 30, "two".to_string(), 3),
            (3, 10, "three".to_string(), 1),
            (3, 20, "three".to_string(), 2),
            (3, 25, "three".to_string(), 2),
            (3, 30, "three".to_string(), 3),
            (4, 10, "four".to_string(), 1),
            (4, 20, "four".to_string(), 2),
            (4, 25, "four".to_string(), 2),
            (4, 30, "four".to_string(), 3),
        ]
    );

    fs::remove_dir_all(dir).unwrap();
}
