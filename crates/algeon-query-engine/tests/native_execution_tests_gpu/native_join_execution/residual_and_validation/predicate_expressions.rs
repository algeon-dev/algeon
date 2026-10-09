use std::sync::Arc;

use super::*;

#[test]
fn native_query_executes_join_residual_cast_predicate() {
    let dir = unique_temp_dir("query-engine-native-mixed-join-cast");
    let left_path = dir.join("lineitem.parquet");
    let right_path = dir.join("orders.parquet");
    let runtime = backend_runtime();
    let left_schema = write_join_left_parquet(&runtime, &left_path);
    let right_schema = write_join_right_parquet(&runtime, &right_path);
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("qty", DataType::Int32, false),
        Field::new("category", DataType::Utf8, false),
    ]));
    let residual = Expr::binary(
        Expr::cast(Expr::input_column(0, "qty", "qty"), DataType::Float64),
        BinaryOp::Gt,
        Expr::binary(
            Expr::cast(Expr::input_column(1, "order_id", "right_order_id"), DataType::Float64),
            BinaryOp::Multiply,
            Expr::literal(LiteralValue::Float64(10.0)),
        ),
    );

    let mut builder = QueryPlanBuilder::new();
    let left =
        builder.add_source(SourceNode::LocalParquet(join_left_source(&left_path, &left_schema, &["order_id", "qty"])));
    let right = builder.add_source(SourceNode::LocalParquet(join_right_source(
        &right_path,
        &right_schema,
        &["order_id", "category"],
    )));
    let join = builder
        .add_join(
            left,
            right,
            JoinSpec::new(JoinType::Inner, vec![join_order_id_key()])
                .with_residual_predicate(residual)
                .with_output_projection(vec![
                    JoinOutputColumn::left(ColumnRef::input(0, "qty", "qty"), "qty"),
                    JoinOutputColumn::right(ColumnRef::input(1, "category", "category"), "category"),
                ])
                .with_build_side(JoinSide::Right),
            Arc::clone(&output_schema),
        )
        .unwrap();
    let plan = builder.finish_with_host_sink(join).unwrap();

    let context = backend_context();
    let result = execute_query_plan(&context, &plan).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 1);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[25]);
    assert_eq!(string_array(batch, 1).iter().collect::<Vec<_>>(), vec![Some("two")]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_query_executes_join_residual_decimal_to_float_cast_predicate() {
    let dir = unique_temp_dir("query-engine-native-mixed-join-decimal-cast");
    let left_path = dir.join("lineitem.parquet");
    let right_path = dir.join("orders.parquet");
    let runtime = backend_runtime();
    let left_schema = write_join_left_decimal_parquet(&runtime, &left_path);
    let right_schema = write_join_right_parquet(&runtime, &right_path);
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("amount", DataType::Decimal128(15, 2), false),
        Field::new("category", DataType::Utf8, false),
    ]));
    let residual = Expr::binary(
        Expr::cast(Expr::input_column(0, "amount", "amount"), DataType::Float64),
        BinaryOp::Eq,
        Expr::binary(
            Expr::cast(Expr::input_column(1, "order_id", "right_order_id"), DataType::Float64),
            BinaryOp::Multiply,
            Expr::literal(LiteralValue::Float64(100.0)),
        ),
    );

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(join_left_source(
        &left_path,
        &left_schema,
        &["order_id", "amount"],
    )));
    let right = builder.add_source(SourceNode::LocalParquet(join_right_source(
        &right_path,
        &right_schema,
        &["order_id", "category"],
    )));
    let join = builder
        .add_join(
            left,
            right,
            JoinSpec::new(JoinType::Inner, vec![join_order_id_key()])
                .with_residual_predicate(residual)
                .with_output_projection(vec![
                    JoinOutputColumn::left(ColumnRef::input(0, "amount", "amount"), "amount"),
                    JoinOutputColumn::right(ColumnRef::input(1, "category", "category"), "category"),
                ])
                .with_build_side(JoinSide::Right),
            Arc::clone(&output_schema),
        )
        .unwrap();
    let plan = builder.finish_with_host_sink(join).unwrap();

    let context = backend_context();
    let result = execute_query_plan(&context, &plan).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 2);
    assert_eq!(decimal128_array(batch, 0).values().as_ref(), &[20_000, 30_000]);
    assert_eq!(string_array(batch, 1).iter().collect::<Vec<_>>(), vec![Some("two"), Some("three")]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_query_executes_join_residual_literal_in_list_predicate() {
    let dir = unique_temp_dir("query-engine-native-join-residual-in-list");
    let left_path = dir.join("lineitem.parquet");
    let right_path = dir.join("orders.parquet");
    let runtime = backend_runtime();
    let left_schema = write_join_left_parquet(&runtime, &left_path);
    let right_schema = write_join_right_parquet(&runtime, &right_path);
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("qty", DataType::Int32, false),
        Field::new("category", DataType::Utf8, false),
    ]));

    let residual = Expr::binary(
        Expr::in_list(
            Expr::input_column(0, "qty", "qty"),
            vec![Expr::literal(LiteralValue::Int32(20)), Expr::literal(LiteralValue::Int32(25))],
        ),
        BinaryOp::And,
        Expr::in_list(
            Expr::input_column(1, "category", "category"),
            vec![
                Expr::literal(LiteralValue::Utf8("two".to_string())),
                Expr::literal(LiteralValue::Utf8("four".to_string())),
            ],
        ),
    );

    let mut builder = QueryPlanBuilder::new();
    let left =
        builder.add_source(SourceNode::LocalParquet(join_left_source(&left_path, &left_schema, &["order_id", "qty"])));
    let right = builder.add_source(SourceNode::LocalParquet(join_right_source(
        &right_path,
        &right_schema,
        &["order_id", "category"],
    )));
    let join = builder
        .add_join(
            left,
            right,
            JoinSpec::new(JoinType::Inner, vec![join_order_id_key()])
                .with_residual_predicate(residual)
                .with_output_projection(vec![
                    JoinOutputColumn::left(ColumnRef::input(0, "qty", "qty"), "qty"),
                    JoinOutputColumn::right(ColumnRef::input(1, "category", "category"), "category"),
                ])
                .with_null_equality(JoinNullEquality::NullsDoNotMatch)
                .with_build_side(JoinSide::Right),
            Arc::clone(&output_schema),
        )
        .unwrap();
    let sort = builder.add_sort(join, SortSpec::new(vec![SortExpr::new(Expr::column("qty"), false, false)])).unwrap();
    let plan = builder.finish_with_host_sink(sort).unwrap();

    let context = backend_context();
    let result = execute_query_plan(&context, &plan).unwrap();
    let join_metrics = result
        .metrics()
        .operators
        .iter()
        .find_map(|operator| operator.join.as_ref())
        .expect("join metrics should be recorded");
    assert!(join_metrics.residual_predicate);
    assert_eq!(join_metrics.output_rows, 2);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[20, 25]);
    assert_eq!(string_array(batch, 1).iter().collect::<Vec<_>>(), vec![Some("two"), Some("two")]);

    fs::remove_dir_all(dir).unwrap();
}
