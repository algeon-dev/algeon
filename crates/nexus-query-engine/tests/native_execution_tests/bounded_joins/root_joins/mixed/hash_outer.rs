use super::*;
#[test]
fn bounded_native_query_executes_root_left_hash_join_with_unmatched_probe_rows() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let dir = unique_temp_dir("nexus-query-engine-bounded-root-left-hash-join");
    let left_path = dir.join("lineitem.parquet");
    let right_path = dir.join("orders.parquet");
    let left_schema = write_join_left_parquet(&runtime, &left_path);
    let right_schema = write_join_right_parquet(&runtime, &right_path);
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
        Field::new("category", DataType::Utf8, true),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let left =
        builder.add_source(SourceNode::LocalParquet(join_left_source(&left_path, &left_schema, &["order_id", "qty"])));
    let right = builder.add_source(SourceNode::LocalParquet(join_right_source(
        &right_path,
        &right_schema,
        &["order_id", "category"],
    )));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(JoinType::Left, vec![join_order_id_key()])
            .with_output_projection(vec![
                JoinOutputColumn::left(nexus_query_engine::ColumnRef::input(0, "order_id", "order_id"), "order_id"),
                JoinOutputColumn::left(nexus_query_engine::ColumnRef::input(0, "qty", "qty"), "qty"),
                JoinOutputColumn::right(nexus_query_engine::ColumnRef::input(1, "category", "category"), "category"),
            ])
            .with_null_equality(JoinNullEquality::NullsDoNotMatch)
            .with_build_side(JoinSide::Right),
        Arc::clone(&output_schema),
    );
    let plan = builder.finish_with_host_sink(join);
    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 1024 * 1024,
            max_source_chunk_bytes: 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .build()
        .unwrap();

    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.phase_c_memory_control_policy.as_str(), "bounded");
    assert_eq!(metrics.reservation_denied_count, 0);
    assert_eq!(metrics.hash_join_build_count, 1);
    let join_metrics =
        metrics.operators.iter().find_map(|operator| operator.join.as_ref()).expect("join metrics should be recorded");
    assert_eq!(join_metrics.join_type, "left");
    assert_eq!(join_metrics.build_side, "right");
    assert_eq!(join_metrics.probe_side, "left");
    assert_eq!(join_metrics.output_rows, 4);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 4);
    assert_eq!(batch.num_columns(), 3);

    let mut rows: Vec<(i32, i32, Option<String>)> = (0..batch.num_rows())
        .map(|row| {
            let order_id = int32_array(batch, 0).value(row);
            let qty = int32_array(batch, 1).value(row);
            let category = string_array(batch, 2);
            let category = if category.is_null(row) { None } else { Some(category.value(row).to_string()) };
            (order_id, qty, category)
        })
        .collect();
    rows.sort();
    assert_eq!(
        rows,
        vec![
            (1, 10, None),
            (2, 20, Some("two".to_string())),
            (2, 25, Some("two".to_string())),
            (3, 30, Some("three".to_string())),
        ]
    );

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_executes_root_full_hash_join_with_symmetric_unmatched_rows() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let dir = unique_temp_dir("nexus-query-engine-bounded-root-full-hash-join");
    let left_path = dir.join("lineitem.parquet");
    let right_path = dir.join("orders.parquet");
    let left_schema = write_join_left_parquet(&runtime, &left_path);
    let right_schema = write_join_right_parquet(&runtime, &right_path);
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("left_order_id", DataType::Int32, true),
        Field::new("right_order_id", DataType::Int32, true),
        Field::new("qty", DataType::Int32, true),
        Field::new("category", DataType::Utf8, true),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let left =
        builder.add_source(SourceNode::LocalParquet(join_left_source(&left_path, &left_schema, &["order_id", "qty"])));
    let right = builder.add_source(SourceNode::LocalParquet(join_right_source(
        &right_path,
        &right_schema,
        &["order_id", "category"],
    )));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(JoinType::Full, vec![join_order_id_key()])
            .with_output_projection(vec![
                JoinOutputColumn::left(
                    nexus_query_engine::ColumnRef::input(0, "order_id", "left_order_id"),
                    "left_order_id",
                ),
                JoinOutputColumn::right(
                    nexus_query_engine::ColumnRef::input(1, "order_id", "right_order_id"),
                    "right_order_id",
                ),
                JoinOutputColumn::left(nexus_query_engine::ColumnRef::input(0, "qty", "qty"), "qty"),
                JoinOutputColumn::right(nexus_query_engine::ColumnRef::input(1, "category", "category"), "category"),
            ])
            .with_null_equality(JoinNullEquality::NullsDoNotMatch)
            .with_build_side(JoinSide::Right),
        Arc::clone(&output_schema),
    );
    let plan = builder.finish_with_host_sink(join);
    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 1024 * 1024,
            max_source_chunk_bytes: 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .build()
        .unwrap();

    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.phase_c_memory_control_policy.as_str(), "bounded");
    assert_eq!(metrics.reservation_denied_count, 0);
    assert_eq!(metrics.hash_join_build_count, 1);
    let join_metrics =
        metrics.operators.iter().find_map(|operator| operator.join.as_ref()).expect("join metrics should be recorded");
    assert_eq!(join_metrics.join_type, "full");
    assert_eq!(join_metrics.build_side, "right");
    assert_eq!(join_metrics.probe_side, "left");
    assert_eq!(join_metrics.output_rows, 5);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 5);
    assert_eq!(batch.num_columns(), 4);

    let left_order_id = int32_array(batch, 0);
    let right_order_id = int32_array(batch, 1);
    let qty = int32_array(batch, 2);
    let category = string_array(batch, 3);
    type Row = (Option<i32>, Option<i32>, Option<i32>, Option<String>);
    let mut rows: Vec<Row> = (0..batch.num_rows())
        .map(|row| {
            (
                if left_order_id.is_null(row) { None } else { Some(left_order_id.value(row)) },
                if right_order_id.is_null(row) { None } else { Some(right_order_id.value(row)) },
                if qty.is_null(row) { None } else { Some(qty.value(row)) },
                if category.is_null(row) { None } else { Some(category.value(row).to_string()) },
            )
        })
        .collect();
    rows.sort();
    assert_eq!(
        rows,
        vec![
            (None, Some(4), None, Some("four".to_string())),
            (Some(1), None, Some(10), None),
            (Some(2), Some(2), Some(20), Some("two".to_string())),
            (Some(2), Some(2), Some(25), Some("two".to_string())),
            (Some(3), Some(3), Some(30), Some("three".to_string())),
        ]
    );

    fs::remove_dir_all(dir).unwrap();
}
