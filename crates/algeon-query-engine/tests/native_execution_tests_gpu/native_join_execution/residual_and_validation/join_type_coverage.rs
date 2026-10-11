use std::sync::Arc;

use super::*;

#[test]
fn native_query_executes_conditional_non_equi_inner_join() {
    let dir = unique_temp_dir("query-engine-native-conditional-inner-join");
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
    let join = builder
        .add_join(
            left,
            right,
            JoinSpec::new(JoinType::Inner, vec![])
                .with_residual_predicate(Expr::binary(
                    Expr::input_column(0, "qty", "qty"),
                    BinaryOp::Lt,
                    right_order_id_times_ten(),
                ))
                .with_output_projection(vec![
                    JoinOutputColumn::left(ColumnRef::input(0, "qty", "qty"), "qty"),
                    JoinOutputColumn::right(ColumnRef::input(1, "order_id", "right_order_id"), "right_order_id"),
                    JoinOutputColumn::right(ColumnRef::input(1, "category", "category"), "category"),
                ]),
            Arc::clone(&output_schema),
        )
        .unwrap();
    let sort = builder
        .add_sort(
            join,
            SortSpec::new(vec![
                SortExpr::new(Expr::column("qty"), false, false),
                SortExpr::new(Expr::column("right_order_id"), false, false),
            ]),
        )
        .unwrap();
    let plan = builder.finish_with_host_sink(sort).unwrap();

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
