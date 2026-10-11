use std::sync::Arc;

use super::*;

#[test]
fn native_query_executes_cross_join_with_explicit_output_projection() {
    let dir = unique_temp_dir("query-engine-native-cross-join");
    let left_path = dir.join("lineitem.parquet");
    let right_path = dir.join("orders.parquet");
    let runtime = backend_runtime();
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
    let join = builder
        .add_join(
            left,
            right,
            JoinSpec::new(JoinType::Cross, vec![]).with_output_projection(vec![
                JoinOutputColumn::left(ColumnRef::input(0, "qty", "qty"), "qty"),
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
                SortExpr::new(Expr::column("category"), false, false),
            ]),
        )
        .unwrap();
    let plan = builder.finish_with_host_sink(sort).unwrap();

    let context = backend_context();
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
