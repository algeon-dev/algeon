use std::sync::Arc;

use super::*;

#[test]
fn native_query_executes_conditional_non_equi_inner_join() {
    let dir = unique_temp_dir("nexus-query-engine-native-conditional-inner-join");
    let left_path = dir.join("lineitem.parquet");
    let right_path = dir.join("orders.parquet");
    let runtime = backend_runtime();
    let left_schema = write_join_left_parquet(&runtime, &left_path);
    let right_schema = write_join_right_parquet(&runtime, &right_path);
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("qty", DataType::Int32, false),
        Field::new("right_order_id", DataType::Int32, false),
        Field::new("category", DataType::Utf8, false),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(join_left_source(&left_path, &left_schema, &["qty"])));
    let right = builder.add_source(SourceNode::LocalParquet(join_right_source(
        &right_path,
        &right_schema,
        &["order_id", "category"],
    )));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(JoinType::Inner, vec![])
            .with_residual_predicate(Expr::binary(
                Expr::input_column(0, "qty", "qty"),
                BinaryOp::Lt,
                right_order_id_times_ten(),
            ))
            .with_output_projection(vec![
                JoinOutputColumn::left(nexus_query_engine::expr::ColumnRef::input(0, "qty", "qty"), "qty"),
                JoinOutputColumn::right(
                    nexus_query_engine::expr::ColumnRef::input(1, "order_id", "right_order_id"),
                    "right_order_id",
                ),
                JoinOutputColumn::right(
                    nexus_query_engine::expr::ColumnRef::input(1, "category", "category"),
                    "category",
                ),
            ]),
        Arc::clone(&output_schema),
    );
    let sort = builder.add_sort(
        join,
        SortSpec::new(vec![
            SortExpr::new(Expr::column("qty"), false, false),
            SortExpr::new(Expr::column("right_order_id"), false, false),
        ]),
        Arc::clone(&output_schema),
    );
    let plan = builder.finish_with_host_sink(sort);

    let context = backend_context();
    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.hash_join_build_count, 0);
    let join_metrics =
        metrics.operators.iter().find_map(|operator| operator.join.as_ref()).expect("join metrics should be recorded");
    assert_eq!(join_metrics.join_type, "inner");
    assert_eq!(join_metrics.key_count, 0);
    assert!(join_metrics.residual_predicate);
    assert_eq!(join_metrics.output_rows, 8);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[10, 10, 10, 20, 20, 25, 25, 30]);
    assert_eq!(int32_array(batch, 1).values().as_ref(), &[2, 3, 4, 3, 4, 3, 4, 4]);
    assert_eq!(
        string_array(batch, 2).iter().collect::<Vec<_>>(),
        vec![
            Some("two"),
            Some("three"),
            Some("four"),
            Some("three"),
            Some("four"),
            Some("three"),
            Some("four"),
            Some("four"),
        ]
    );

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_query_executes_left_join_with_equi_key_and_residual_predicate() {
    let dir = unique_temp_dir("nexus-query-engine-native-mixed-left-join");
    let left_path = dir.join("lineitem.parquet");
    let right_path = dir.join("orders.parquet");
    let runtime = backend_runtime();
    let left_schema = write_join_left_parquet(&runtime, &left_path);
    let right_schema = write_join_right_parquet(&runtime, &right_path);
    let output_schema = Arc::new(Schema::new(vec![
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
            .with_residual_predicate(qty_gt_right_order_id_times_ten())
            .with_output_projection(vec![
                JoinOutputColumn::left(nexus_query_engine::expr::ColumnRef::input(0, "qty", "qty"), "qty"),
                JoinOutputColumn::right(
                    nexus_query_engine::expr::ColumnRef::input(1, "category", "category"),
                    "category",
                ),
            ])
            .with_build_side(JoinSide::Right),
        Arc::clone(&output_schema),
    );
    let sort = builder.add_sort(
        join,
        SortSpec::new(vec![SortExpr::new(Expr::column("qty"), false, false)]),
        Arc::clone(&output_schema),
    );
    let plan = builder.finish_with_host_sink(sort);

    let context = backend_context();
    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert!(metrics.mixed_join_size_data_bytes > 0);
    let join_metrics =
        metrics.operators.iter().find_map(|operator| operator.join.as_ref()).expect("join metrics should be recorded");
    assert_eq!(join_metrics.join_type, "left");
    assert_eq!(join_metrics.key_count, 1);
    assert!(join_metrics.residual_predicate);
    assert_eq!(join_metrics.mixed_size_data_bytes, metrics.mixed_join_size_data_bytes);
    assert_eq!(join_metrics.output_rows, 4);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[10, 20, 25, 30]);
    let category = string_array(batch, 1);
    assert!(category.is_null(0));
    assert!(category.is_null(1));
    assert_eq!(category.value(2), "two");
    assert!(category.is_null(3));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_query_executes_full_join_with_equi_key_and_residual_predicate() {
    let dir = unique_temp_dir("nexus-query-engine-native-mixed-full-join");
    let left_path = dir.join("lineitem.parquet");
    let right_path = dir.join("orders.parquet");
    let runtime = backend_runtime();
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
            .with_residual_predicate(qty_gt_right_order_id_times_ten())
            .with_output_projection(vec![
                JoinOutputColumn::left(
                    nexus_query_engine::expr::ColumnRef::input(0, "order_id", "left_order_id"),
                    "left_order_id",
                ),
                JoinOutputColumn::right(
                    nexus_query_engine::expr::ColumnRef::input(1, "order_id", "right_order_id"),
                    "right_order_id",
                ),
                JoinOutputColumn::left(nexus_query_engine::expr::ColumnRef::input(0, "qty", "qty"), "qty"),
                JoinOutputColumn::right(
                    nexus_query_engine::expr::ColumnRef::input(1, "category", "category"),
                    "category",
                ),
            ])
            .with_build_side(JoinSide::Right),
        Arc::clone(&output_schema),
    );
    let sort = builder.add_sort(
        join,
        SortSpec::new(vec![
            SortExpr::new(Expr::column("left_order_id"), false, false),
            SortExpr::new(Expr::column("qty"), false, false),
            SortExpr::new(Expr::column("right_order_id"), false, false),
        ]),
        Arc::clone(&output_schema),
    );
    let plan = builder.finish_with_host_sink(sort);

    let context = backend_context();
    let result = execute_query_plan(&context, &plan).unwrap();
    let join_metrics = result
        .metrics()
        .operators
        .iter()
        .find_map(|operator| operator.join.as_ref())
        .expect("join metrics should be recorded");
    assert_eq!(join_metrics.join_type, "full");
    assert_eq!(join_metrics.key_count, 1);
    assert!(join_metrics.residual_predicate);
    assert_eq!(join_metrics.output_rows, 6);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    let left_order_id = int32_array(batch, 0);
    assert_eq!(left_order_id.value(0), 1);
    assert_eq!(left_order_id.value(1), 2);
    assert_eq!(left_order_id.value(2), 2);
    assert_eq!(left_order_id.value(3), 3);
    assert!(left_order_id.is_null(4));
    assert!(left_order_id.is_null(5));

    let right_order_id = int32_array(batch, 1);
    assert!(right_order_id.is_null(0));
    assert!(right_order_id.is_null(1));
    assert_eq!(right_order_id.value(2), 2);
    assert!(right_order_id.is_null(3));
    assert_eq!(right_order_id.value(4), 3);
    assert_eq!(right_order_id.value(5), 4);

    let qty = int32_array(batch, 2);
    assert_eq!(qty.value(0), 10);
    assert_eq!(qty.value(1), 20);
    assert_eq!(qty.value(2), 25);
    assert_eq!(qty.value(3), 30);
    assert!(qty.is_null(4));
    assert!(qty.is_null(5));

    let category = string_array(batch, 3);
    assert!(category.is_null(0));
    assert!(category.is_null(1));
    assert_eq!(category.value(2), "two");
    assert!(category.is_null(3));
    assert_eq!(category.value(4), "three");
    assert_eq!(category.value(5), "four");

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_query_executes_left_semi_and_anti_joins_with_equi_key_and_residual_predicate() {
    let dir = unique_temp_dir("nexus-query-engine-native-mixed-semi-anti-join");
    let left_path = dir.join("lineitem.parquet");
    let right_path = dir.join("orders.parquet");
    let runtime = backend_runtime();
    let left_schema = write_join_left_parquet(&runtime, &left_path);
    let right_schema = write_join_right_parquet(&runtime, &right_path);

    let semi = execute_left_only_residual_join(
        &runtime,
        &left_path,
        &right_path,
        &left_schema,
        &right_schema,
        JoinType::LeftSemi,
    );
    let anti = execute_left_only_residual_join(
        &runtime,
        &left_path,
        &right_path,
        &left_schema,
        &right_schema,
        JoinType::LeftAnti,
    );

    let QueryOutput::HostArrow(semi_batch) = semi.output() else {
        panic!("left semi residual join should return host output");
    };
    assert_eq!(int32_array(semi_batch, 0).values().as_ref(), &[25]);
    let semi_metrics = semi
        .metrics()
        .operators
        .iter()
        .find_map(|operator| operator.join.as_ref())
        .expect("join metrics should be recorded");
    assert_eq!(semi_metrics.join_type, "left_semi");
    assert_eq!(semi_metrics.key_count, 1);
    assert!(semi_metrics.residual_predicate);

    let QueryOutput::HostArrow(anti_batch) = anti.output() else {
        panic!("left anti residual join should return host output");
    };
    assert_eq!(int32_array(anti_batch, 0).values().as_ref(), &[10, 20, 30]);
    let anti_metrics = anti
        .metrics()
        .operators
        .iter()
        .find_map(|operator| operator.join.as_ref())
        .expect("join metrics should be recorded");
    assert_eq!(anti_metrics.join_type, "left_anti");
    assert_eq!(anti_metrics.key_count, 1);
    assert!(anti_metrics.residual_predicate);

    fs::remove_dir_all(dir).unwrap();
}
