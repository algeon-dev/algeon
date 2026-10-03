use std::{path::Path, sync::Arc};

use query_engine::expr::ColumnRef;

use super::*;

#[test]
fn bounded_native_query_executes_root_keyless_conditional_join_with_ordered_reservations() {
    let runtime = backend_runtime();
    let dir = unique_temp_dir("query-engine-bounded-root-conditional-join");
    let left_path = dir.join("lineitem.parquet");
    let right_path = dir.join("orders.parquet");
    let left_schema = write_join_left_parquet(&runtime, &left_path);
    let right_schema = write_join_right_parquet(&runtime, &right_path);
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("qty", DataType::Int32, false),
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
                .with_residual_predicate(qty_gt_right_order_id_times_ten())
                .with_output_projection(vec![
                    JoinOutputColumn::left(ColumnRef::input(0, "qty", "qty"), "qty"),
                    JoinOutputColumn::right(ColumnRef::input(1, "category", "category"), "category"),
                ]),
            Arc::clone(&output_schema),
        )
        .unwrap();
    let plan = builder.finish_with_host_sink(join).unwrap();
    assert_bounded_pipeline_admitted(&plan);

    let context = backend_context_with_grant(1024 * 1024);

    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.parquet_files, 2);
    assert_eq!(metrics.hash_join_build_count, 0);
    assert_eq!(metrics.hash_join_hash_table_bytes, 0);
    let join_metrics =
        metrics.operators.iter().find_map(|operator| operator.join.as_ref()).expect("join metrics should be recorded");
    assert_eq!(join_metrics.join_type, "inner");
    assert_eq!(join_metrics.key_count, 0);
    assert!(join_metrics.residual_predicate);
    assert_eq!(join_metrics.hash_table_bytes, 0);
    assert!(join_metrics.gather_map_bytes > 0);
    assert_eq!(join_metrics.output_rows, 2);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 2);
    assert_eq!(batch.num_columns(), 2);
    let mut rows = (0..batch.num_rows())
        .map(|row| (int32_array(batch, 0).value(row), string_array(batch, 1).value(row).to_string()))
        .collect::<Vec<_>>();
    rows.sort();
    assert_eq!(rows, vec![(25, "two".to_string()), (30, "two".to_string())]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_executes_root_keyless_conditional_left_join_with_unmatched_left_rows() {
    let runtime = backend_runtime();
    let dir = unique_temp_dir("query-engine-bounded-root-conditional-left-join");
    let left_path = dir.join("lineitem.parquet");
    let right_path = dir.join("orders.parquet");
    let left_schema = write_join_left_parquet(&runtime, &left_path);
    let right_schema = write_join_right_parquet(&runtime, &right_path);
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("qty", DataType::Int32, false),
        Field::new("category", DataType::Utf8, true),
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
            JoinSpec::new(JoinType::Left, vec![])
                .with_residual_predicate(qty_gt_right_order_id_times_ten())
                .with_output_projection(vec![
                    JoinOutputColumn::left(ColumnRef::input(0, "qty", "qty"), "qty"),
                    JoinOutputColumn::right(ColumnRef::input(1, "category", "category"), "category"),
                ])
                .with_build_side(JoinSide::Right),
            Arc::clone(&output_schema),
        )
        .unwrap();
    let plan = builder.finish_with_host_sink(join).unwrap();
    assert_bounded_pipeline_admitted(&plan);

    let context = bounded_join_execution_context(&runtime);
    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.hash_join_build_count, 0);
    let join_metrics =
        metrics.operators.iter().find_map(|operator| operator.join.as_ref()).expect("join metrics should be recorded");
    assert_eq!(join_metrics.join_type, "left");
    assert_eq!(join_metrics.execution_strategy, "bounded_keyless_conditional_join");
    assert_eq!(join_metrics.key_count, 0);
    assert!(join_metrics.residual_predicate);
    assert_eq!(join_metrics.hash_table_bytes, 0);
    assert!(join_metrics.gather_map_bytes > 0);
    assert_eq!(join_metrics.output_rows, 4);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 4);
    let category = string_array(batch, 1);
    let mut rows = (0..batch.num_rows())
        .map(|row| {
            let category = if category.is_null(row) { None } else { Some(category.value(row).to_string()) };
            (int32_array(batch, 0).value(row), category)
        })
        .collect::<Vec<_>>();
    rows.sort();
    assert_eq!(rows, vec![(10, None), (20, None), (25, Some("two".to_string())), (30, Some("two".to_string())),]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_executes_root_keyless_conditional_full_join_with_retained_size_data() {
    let runtime = backend_runtime();
    let dir = unique_temp_dir("query-engine-bounded-root-conditional-full-join");
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
    let join = builder
        .add_join(
            left,
            right,
            JoinSpec::new(JoinType::Full, vec![])
                .with_residual_predicate(qty_gt_right_order_id_times_ten())
                .with_output_projection(vec![
                    JoinOutputColumn::left(ColumnRef::input(0, "order_id", "left_order_id"), "left_order_id"),
                    JoinOutputColumn::right(ColumnRef::input(1, "order_id", "right_order_id"), "right_order_id"),
                    JoinOutputColumn::left(ColumnRef::input(0, "qty", "qty"), "qty"),
                    JoinOutputColumn::right(ColumnRef::input(1, "category", "category"), "category"),
                ]),
            Arc::clone(&output_schema),
        )
        .unwrap();
    let plan = builder.finish_with_host_sink(join).unwrap();
    assert_bounded_pipeline_admitted(&plan);

    let context = bounded_join_execution_context(&runtime);
    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.hash_join_build_count, 0);
    assert!(metrics.mixed_join_size_data_bytes > 0);
    let join_metrics =
        metrics.operators.iter().find_map(|operator| operator.join.as_ref()).expect("join metrics should be recorded");
    assert_eq!(join_metrics.join_type, "full");
    assert_eq!(join_metrics.execution_strategy, "bounded_keyless_conditional_join");
    assert_eq!(join_metrics.key_count, 0);
    assert!(join_metrics.residual_predicate);
    assert_eq!(join_metrics.hash_table_bytes, 0);
    assert_eq!(join_metrics.mixed_size_data_bytes, metrics.mixed_join_size_data_bytes);
    assert_eq!(join_metrics.output_rows, 6);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    let left_order_id = int32_array(batch, 0);
    let right_order_id = int32_array(batch, 1);
    let qty = int32_array(batch, 2);
    let category = string_array(batch, 3);
    let mut rows = (0..batch.num_rows())
        .map(|row| {
            (
                if left_order_id.is_null(row) { None } else { Some(left_order_id.value(row)) },
                if right_order_id.is_null(row) { None } else { Some(right_order_id.value(row)) },
                if qty.is_null(row) { None } else { Some(qty.value(row)) },
                if category.is_null(row) { None } else { Some(category.value(row).to_string()) },
            )
        })
        .collect::<Vec<_>>();
    rows.sort();
    assert_eq!(
        rows,
        vec![
            (None, Some(3), None, Some("three".to_string())),
            (None, Some(4), None, Some("four".to_string())),
            (Some(1), None, Some(10), None),
            (Some(2), None, Some(20), None),
            (Some(2), Some(2), Some(25), Some("two".to_string())),
            (Some(3), Some(2), Some(30), Some("two".to_string())),
        ]
    );

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_executes_keyless_predicate_free_left_join_as_outer_cross() {
    let runtime = backend_runtime();
    let dir = unique_temp_dir("query-engine-bounded-keyless-left-outer-cross");
    let left_path = dir.join("lineitem.parquet");
    let right_path = dir.join("orders.parquet");
    let left_schema = write_join_left_parquet(&runtime, &left_path);
    let right_schema = write_join_right_parquet(&runtime, &right_path);
    let plan = keyless_outer_cross_join_plan(
        &left_path,
        &right_path,
        &left_schema,
        &right_schema,
        JoinType::Left,
        false,
        false,
    );
    assert_bounded_pipeline_admitted(&plan);

    let context = bounded_join_execution_context(&runtime);
    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    let join_metrics =
        metrics.operators.iter().find_map(|operator| operator.join.as_ref()).expect("join metrics should be recorded");
    assert_eq!(join_metrics.join_type, "left");
    assert_eq!(join_metrics.execution_strategy, "bounded_keyless_outer_cross_join");
    assert_eq!(join_metrics.key_count, 0);
    assert!(!join_metrics.residual_predicate);
    assert_eq!(join_metrics.output_rows, 12);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 12);
    let mut rows = (0..batch.num_rows())
        .map(|row| (int32_array(batch, 0).value(row), string_array(batch, 1).value(row).to_string()))
        .collect::<Vec<_>>();
    rows.sort();
    assert_eq!(
        rows,
        vec![
            (10, "four".to_string()),
            (10, "three".to_string()),
            (10, "two".to_string()),
            (20, "four".to_string()),
            (20, "three".to_string()),
            (20, "two".to_string()),
            (25, "four".to_string()),
            (25, "three".to_string()),
            (25, "two".to_string()),
            (30, "four".to_string()),
            (30, "three".to_string()),
            (30, "two".to_string()),
        ]
    );

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_null_extends_keyless_predicate_free_left_join_when_right_is_empty() {
    let runtime = backend_runtime();
    let dir = unique_temp_dir("query-engine-bounded-keyless-left-outer-cross-empty-right");
    let left_path = dir.join("lineitem.parquet");
    let right_path = dir.join("orders.parquet");
    let left_schema = write_join_left_parquet(&runtime, &left_path);
    let right_schema = write_join_right_parquet(&runtime, &right_path);
    let plan = keyless_outer_cross_join_plan(
        &left_path,
        &right_path,
        &left_schema,
        &right_schema,
        JoinType::Left,
        false,
        true,
    );
    assert_bounded_pipeline_admitted(&plan);

    let context = bounded_join_execution_context(&runtime);
    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    let join_metrics =
        metrics.operators.iter().find_map(|operator| operator.join.as_ref()).expect("join metrics should be recorded");
    assert_eq!(join_metrics.join_type, "left");
    assert_eq!(join_metrics.execution_strategy, "bounded_keyless_outer_cross_join");
    assert_eq!(join_metrics.output_rows, 4);
    assert!(join_metrics.gather_map_bytes > 0);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[10, 20, 25, 30]);
    let category = string_array(batch, 1);
    assert!((0..batch.num_rows()).all(|row| category.is_null(row)));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_null_extends_keyless_predicate_free_full_join_when_left_is_empty() {
    let runtime = backend_runtime();
    let dir = unique_temp_dir("query-engine-bounded-keyless-full-outer-cross-empty-left");
    let left_path = dir.join("lineitem.parquet");
    let right_path = dir.join("orders.parquet");
    let left_schema = write_join_left_parquet(&runtime, &left_path);
    let right_schema = write_join_right_parquet(&runtime, &right_path);
    let plan = keyless_outer_cross_join_plan(
        &left_path,
        &right_path,
        &left_schema,
        &right_schema,
        JoinType::Full,
        true,
        false,
    );
    assert_bounded_pipeline_admitted(&plan);

    let context = bounded_join_execution_context(&runtime);
    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    let join_metrics =
        metrics.operators.iter().find_map(|operator| operator.join.as_ref()).expect("join metrics should be recorded");
    assert_eq!(join_metrics.join_type, "full");
    assert_eq!(join_metrics.execution_strategy, "bounded_keyless_outer_cross_join");
    assert_eq!(join_metrics.output_rows, 3);
    assert!(join_metrics.gather_map_bytes > 0);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    let qty = int32_array(batch, 0);
    assert!((0..batch.num_rows()).all(|row| qty.is_null(row)));
    let mut categories = string_array(batch, 1).iter().map(|value| value.unwrap().to_string()).collect::<Vec<_>>();
    categories.sort();
    assert_eq!(categories, vec!["four".to_string(), "three".to_string(), "two".to_string()]);

    fs::remove_dir_all(dir).unwrap();
}

fn keyless_outer_cross_join_plan(
    left_path: &Path,
    right_path: &Path,
    left_schema: &Arc<Schema>,
    right_schema: &Arc<Schema>,
    join_type: JoinType,
    empty_left: bool,
    empty_right: bool,
) -> QueryPlan {
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("qty", DataType::Int32, join_type == JoinType::Full),
        Field::new("category", DataType::Utf8, true),
    ]));
    let mut builder = QueryPlanBuilder::new();
    let left =
        builder.add_source(SourceNode::LocalParquet(join_left_source(left_path, left_schema, &["order_id", "qty"])));
    let left = if empty_left {
        builder
            .add_filter(left, Expr::binary(Expr::column("qty"), BinaryOp::Lt, Expr::literal(LiteralValue::Int32(0))))
            .unwrap()
    } else {
        left
    };
    let right = builder.add_source(SourceNode::LocalParquet(join_right_source(
        right_path,
        right_schema,
        &["order_id", "category"],
    )));
    let right = if empty_right {
        builder
            .add_filter(
                right,
                Expr::binary(Expr::column("order_id"), BinaryOp::Lt, Expr::literal(LiteralValue::Int32(0))),
            )
            .unwrap()
    } else {
        right
    };
    let join = builder
        .add_join(
            left,
            right,
            JoinSpec::new(join_type.clone(), vec![])
                .with_output_projection(vec![
                    JoinOutputColumn::left(ColumnRef::input(0, "qty", "qty"), "qty"),
                    JoinOutputColumn::right(ColumnRef::input(1, "category", "category"), "category"),
                ])
                .with_build_side(match join_type {
                    JoinType::Left => JoinSide::Right,
                    JoinType::Full => JoinSide::Left,
                    _ => unreachable!("test helper only builds left/full joins"),
                }),
            Arc::clone(&output_schema),
        )
        .unwrap();
    builder.finish_with_host_sink(join).unwrap()
}
