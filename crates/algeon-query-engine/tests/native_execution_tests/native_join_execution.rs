use std::sync::Arc;

use super::*;

#[path = "native_join_execution/cross_projection.rs"]
mod cross_projection;
#[path = "native_join_execution/residual_and_validation.rs"]
mod residual_and_validation;

#[test]
fn native_query_executes_inner_equi_join_with_explicit_output_projection() {
    let dir = unique_temp_dir("query-engine-native-inner-join");
    let left_path = dir.join("lineitem.parquet");
    let right_path = dir.join("orders.parquet");
    let context = backend_context();
    let runtime = Arc::new(context.runtime().unwrap());
    let left_schema = write_join_left_parquet(&runtime, &left_path);
    let right_schema = write_join_right_parquet(&runtime, &right_path);
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
        Field::new("category", DataType::Utf8, false),
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
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                query_engine::expr::ColumnRef::input(0, "order_id", "left_order_id"),
                query_engine::expr::ColumnRef::input(1, "order_id", "right_order_id"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(query_engine::expr::ColumnRef::input(0, "order_id", "order_id"), "order_id"),
            JoinOutputColumn::left(query_engine::expr::ColumnRef::input(0, "qty", "qty"), "qty"),
            JoinOutputColumn::right(query_engine::expr::ColumnRef::input(1, "category", "category"), "category"),
        ])
        .with_null_equality(JoinNullEquality::NullsDoNotMatch)
        .with_build_side(JoinSide::Right),
        Arc::clone(&output_schema),
    );
    let sort = builder.add_sort(
        join,
        SortSpec::new(vec![
            SortExpr::new(Expr::column("order_id"), false, false),
            SortExpr::new(Expr::column("qty"), false, false),
        ]),
        Arc::clone(&output_schema),
    );
    let plan = builder.finish_with_host_sink(sort);

    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.parquet_files, 2);
    assert_eq!(metrics.input_batches, 2);
    assert_eq!(metrics.input_rows, 7);
    assert_eq!(metrics.boundary.host_to_device_uploads, 0);
    assert_eq!(metrics.boundary.device_to_host_materializations, 1);
    let join_metrics =
        metrics.operators.iter().find_map(|operator| operator.join.as_ref()).expect("join metrics should be recorded");
    assert_eq!(join_metrics.join_type, "inner");
    assert_eq!(join_metrics.key_count, 1);
    assert_eq!(join_metrics.build_side, "right");
    assert_eq!(join_metrics.probe_side, "left");
    assert_eq!(join_metrics.build_rows, 3);
    assert_eq!(join_metrics.probe_rows, 4);
    assert!(join_metrics.hash_table_bytes > 0);
    assert_eq!(join_metrics.output_rows, 3);
    assert!(join_metrics.output_bytes > 0);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 3);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[2, 2, 3]);
    assert_eq!(int32_array(batch, 1).values().as_ref(), &[20, 25, 30]);
    assert_eq!(string_array(batch, 2).iter().collect::<Vec<_>>(), vec![Some("two"), Some("two"), Some("three")]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_query_executes_inner_join_when_output_projection_omits_keys() {
    let dir = unique_temp_dir("query-engine-native-inner-join-without-key-output");
    let left_path = dir.join("lineitem.parquet");
    let right_path = dir.join("orders.parquet");
    let context = backend_context();
    let runtime = Arc::new(context.runtime().unwrap());
    let left_schema = write_join_left_parquet(&runtime, &left_path);
    let right_schema = write_join_right_parquet(&runtime, &right_path);
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("qty", DataType::Int32, false),
        Field::new("category", DataType::Utf8, false),
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
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                query_engine::expr::ColumnRef::input(0, "order_id", "left_order_id"),
                query_engine::expr::ColumnRef::input(1, "order_id", "right_order_id"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(query_engine::expr::ColumnRef::input(0, "qty", "qty"), "qty"),
            JoinOutputColumn::right(query_engine::expr::ColumnRef::input(1, "category", "category"), "category"),
        ])
        .with_null_equality(JoinNullEquality::NullsDoNotMatch)
        .with_build_side(JoinSide::Right),
        Arc::clone(&output_schema),
    );
    let sort = builder.add_sort(
        join,
        SortSpec::new(vec![SortExpr::new(Expr::column("qty"), false, false)]),
        Arc::clone(&output_schema),
    );
    let plan = builder.finish_with_host_sink(sort);

    let result = execute_query_plan(&context, &plan).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 3);
    assert_eq!(batch.num_columns(), 2);
    assert_eq!(batch.schema().field(0).name(), "qty");
    assert_eq!(batch.schema().field(1).name(), "category");
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[20, 25, 30]);
    assert_eq!(string_array(batch, 1).iter().collect::<Vec<_>>(), vec![Some("two"), Some("two"), Some("three")]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_query_executes_left_join_with_null_padded_right_projection() {
    let dir = unique_temp_dir("query-engine-native-left-join");
    let left_path = dir.join("lineitem.parquet");
    let right_path = dir.join("orders.parquet");
    let context = backend_context();
    let runtime = Arc::new(context.runtime().unwrap());
    let left_schema = write_join_left_parquet(&runtime, &left_path);
    let right_schema = write_join_right_parquet(&runtime, &right_path);
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("qty", DataType::Int32, false),
        Field::new("category", DataType::Utf8, true),
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
        JoinSpec::new(
            JoinType::Left,
            vec![JoinKey::new(
                query_engine::expr::ColumnRef::input(0, "order_id", "left_order_id"),
                query_engine::expr::ColumnRef::input(1, "order_id", "right_order_id"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(query_engine::expr::ColumnRef::input(0, "qty", "qty"), "qty"),
            JoinOutputColumn::right(query_engine::expr::ColumnRef::input(1, "category", "category"), "category"),
        ])
        .with_null_equality(JoinNullEquality::NullsDoNotMatch)
        .with_build_side(JoinSide::Right),
        Arc::clone(&output_schema),
    );
    let sort = builder.add_sort(
        join,
        SortSpec::new(vec![SortExpr::new(Expr::column("qty"), false, false)]),
        Arc::clone(&output_schema),
    );
    let plan = builder.finish_with_host_sink(sort);

    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    let join_metrics =
        metrics.operators.iter().find_map(|operator| operator.join.as_ref()).expect("join metrics should be recorded");
    assert_eq!(join_metrics.join_type, "left");
    assert_eq!(join_metrics.key_count, 1);
    assert_eq!(join_metrics.build_side, "right");
    assert_eq!(join_metrics.probe_side, "left");
    assert_eq!(join_metrics.build_rows, 3);
    assert_eq!(join_metrics.probe_rows, 4);
    assert_eq!(join_metrics.output_rows, 4);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 4);
    assert_eq!(batch.num_columns(), 2);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[10, 20, 25, 30]);
    let category = string_array(batch, 1);
    assert!(category.is_null(0));
    assert_eq!(category.value(1), "two");
    assert_eq!(category.value(2), "two");
    assert_eq!(category.value(3), "three");

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_query_executes_full_join_with_null_padded_output_projection() {
    let dir = unique_temp_dir("query-engine-native-full-join");
    let left_path = dir.join("lineitem.parquet");
    let right_path = dir.join("orders.parquet");
    let context = backend_context();
    let runtime = Arc::new(context.runtime().unwrap());
    let left_schema = write_join_left_parquet(&runtime, &left_path);
    let right_schema = write_join_right_parquet(&runtime, &right_path);
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("left_order_id", DataType::Int32, true),
        Field::new("right_order_id", DataType::Int32, true),
        Field::new("qty", DataType::Int32, true),
        Field::new("category", DataType::Utf8, true),
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
        JoinSpec::new(
            JoinType::Full,
            vec![JoinKey::new(
                query_engine::expr::ColumnRef::input(0, "order_id", "left_order_id"),
                query_engine::expr::ColumnRef::input(1, "order_id", "right_order_id"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(
                query_engine::expr::ColumnRef::input(0, "order_id", "left_order_id"),
                "left_order_id",
            ),
            JoinOutputColumn::right(
                query_engine::expr::ColumnRef::input(1, "order_id", "right_order_id"),
                "right_order_id",
            ),
            JoinOutputColumn::left(query_engine::expr::ColumnRef::input(0, "qty", "qty"), "qty"),
            JoinOutputColumn::right(query_engine::expr::ColumnRef::input(1, "category", "category"), "category"),
        ])
        .with_null_equality(JoinNullEquality::NullsDoNotMatch)
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

    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    let join_metrics =
        metrics.operators.iter().find_map(|operator| operator.join.as_ref()).expect("join metrics should be recorded");
    assert_eq!(join_metrics.join_type, "full");
    assert_eq!(join_metrics.key_count, 1);
    assert_eq!(join_metrics.output_rows, 5);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 5);
    assert_eq!(batch.num_columns(), 4);

    let left_order_id = int32_array(batch, 0);
    assert_eq!(left_order_id.value(0), 1);
    assert_eq!(left_order_id.value(1), 2);
    assert_eq!(left_order_id.value(2), 2);
    assert_eq!(left_order_id.value(3), 3);
    assert!(left_order_id.is_null(4));

    let right_order_id = int32_array(batch, 1);
    assert!(right_order_id.is_null(0));
    assert_eq!(right_order_id.value(1), 2);
    assert_eq!(right_order_id.value(2), 2);
    assert_eq!(right_order_id.value(3), 3);
    assert_eq!(right_order_id.value(4), 4);

    let qty = int32_array(batch, 2);
    assert_eq!(qty.value(0), 10);
    assert_eq!(qty.value(1), 20);
    assert_eq!(qty.value(2), 25);
    assert_eq!(qty.value(3), 30);
    assert!(qty.is_null(4));

    let category = string_array(batch, 3);
    assert!(category.is_null(0));
    assert_eq!(category.value(1), "two");
    assert_eq!(category.value(2), "two");
    assert_eq!(category.value(3), "three");
    assert_eq!(category.value(4), "four");

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_query_executes_left_semi_join_with_left_only_projection() {
    let dir = unique_temp_dir("query-engine-native-left-semi-join");
    let left_path = dir.join("lineitem.parquet");
    let right_path = dir.join("orders.parquet");
    let context = backend_context();
    let runtime = Arc::new(context.runtime().unwrap());
    let left_schema = write_join_left_parquet(&runtime, &left_path);
    let right_schema = write_join_right_parquet(&runtime, &right_path);
    let output_schema = Arc::new(Schema::new(vec![Field::new("qty", DataType::Int32, false)]));

    let left_source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&left_path).with_file_size_bytes(fs::metadata(&left_path).unwrap().len())],
        Arc::clone(&left_schema),
        vec!["order_id".to_string(), "qty".to_string()],
    );
    let right_source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&right_path).with_file_size_bytes(fs::metadata(&right_path).unwrap().len())],
        Arc::clone(&right_schema),
        vec!["order_id".to_string()],
    );

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(left_source));
    let right = builder.add_source(SourceNode::LocalParquet(right_source));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(
            JoinType::LeftSemi,
            vec![JoinKey::new(
                query_engine::expr::ColumnRef::input(0, "order_id", "left_order_id"),
                query_engine::expr::ColumnRef::input(1, "order_id", "right_order_id"),
            )],
        )
        .with_output_projection(vec![JoinOutputColumn::left(
            query_engine::expr::ColumnRef::input(0, "qty", "qty"),
            "qty",
        )])
        .with_null_equality(JoinNullEquality::NullsDoNotMatch)
        .with_build_side(JoinSide::Right),
        Arc::clone(&output_schema),
    );
    let sort = builder.add_sort(
        join,
        SortSpec::new(vec![SortExpr::new(Expr::column("qty"), false, false)]),
        Arc::clone(&output_schema),
    );
    let plan = builder.finish_with_host_sink(sort);

    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    let join_metrics =
        metrics.operators.iter().find_map(|operator| operator.join.as_ref()).expect("join metrics should be recorded");
    assert_eq!(join_metrics.join_type, "left_semi");
    assert_eq!(join_metrics.output_rows, 3);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 3);
    assert_eq!(batch.num_columns(), 1);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[20, 25, 30]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_query_executes_left_anti_join_with_left_only_projection() {
    let dir = unique_temp_dir("query-engine-native-left-anti-join");
    let left_path = dir.join("lineitem.parquet");
    let right_path = dir.join("orders.parquet");
    let context = backend_context();
    let runtime = Arc::new(context.runtime().unwrap());
    let left_schema = write_join_left_parquet(&runtime, &left_path);
    let right_schema = write_join_right_parquet(&runtime, &right_path);
    let output_schema = Arc::new(Schema::new(vec![Field::new("qty", DataType::Int32, false)]));

    let left_source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&left_path).with_file_size_bytes(fs::metadata(&left_path).unwrap().len())],
        Arc::clone(&left_schema),
        vec!["order_id".to_string(), "qty".to_string()],
    );
    let right_source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&right_path).with_file_size_bytes(fs::metadata(&right_path).unwrap().len())],
        Arc::clone(&right_schema),
        vec!["order_id".to_string()],
    );

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(left_source));
    let right = builder.add_source(SourceNode::LocalParquet(right_source));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(
            JoinType::LeftAnti,
            vec![JoinKey::new(
                query_engine::expr::ColumnRef::input(0, "order_id", "left_order_id"),
                query_engine::expr::ColumnRef::input(1, "order_id", "right_order_id"),
            )],
        )
        .with_output_projection(vec![JoinOutputColumn::left(
            query_engine::expr::ColumnRef::input(0, "qty", "qty"),
            "qty",
        )])
        .with_null_equality(JoinNullEquality::NullsDoNotMatch)
        .with_build_side(JoinSide::Right),
        Arc::clone(&output_schema),
    );
    let plan = builder.finish_with_host_sink(join);

    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    let join_metrics =
        metrics.operators.iter().find_map(|operator| operator.join.as_ref()).expect("join metrics should be recorded");
    assert_eq!(join_metrics.join_type, "left_anti");
    assert_eq!(join_metrics.output_rows, 1);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 1);
    assert_eq!(batch.num_columns(), 1);
    assert_eq!(int32_array(batch, 0).value(0), 10);

    fs::remove_dir_all(dir).unwrap();
}

fn null_aware_anti_join_output_qtys<C: TestExecutionContext + ?Sized>(
    context: &C,
    runtime: &cudf::Runtime,
    dir: &std::path::Path,
    case: &str,
    left_keys: Vec<Option<i32>>,
    right_keys: Vec<Option<i32>>,
) -> Vec<i32> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, true),
        Field::new("qty", DataType::Int32, false),
    ]));
    let write = |path: &std::path::Path, keys: Vec<Option<i32>>| {
        let qtys = (0..keys.len() as i32).collect::<Vec<_>>();
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int32Array::from(keys)), Arc::new(Int32Array::from(qtys))],
        )
        .unwrap();
        DataFrame::from_record_batch(runtime, &batch).unwrap().write_parquet(path).unwrap();
    };
    let left_path = dir.join(format!("{case}-left.parquet"));
    let right_path = dir.join(format!("{case}-right.parquet"));
    write(&left_path, left_keys);
    write(&right_path, right_keys);

    let output_schema = Arc::new(Schema::new(vec![Field::new("qty", DataType::Int32, false)]));
    let left_source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&left_path).with_file_size_bytes(fs::metadata(&left_path).unwrap().len())],
        Arc::clone(&schema),
        vec!["key".to_string(), "qty".to_string()],
    );
    let right_source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&right_path).with_file_size_bytes(fs::metadata(&right_path).unwrap().len())],
        Arc::clone(&schema),
        vec!["key".to_string()],
    );

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(left_source));
    let right = builder.add_source(SourceNode::LocalParquet(right_source));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(
            JoinType::LeftAnti,
            vec![JoinKey::new(
                query_engine::expr::ColumnRef::input(0, "key", "left_key"),
                query_engine::expr::ColumnRef::input(1, "key", "right_key"),
            )],
        )
        .with_output_projection(vec![JoinOutputColumn::left(
            query_engine::expr::ColumnRef::input(0, "qty", "qty"),
            "qty",
        )])
        .with_null_equality(JoinNullEquality::NullsDoNotMatch)
        .with_build_side(JoinSide::Right)
        .with_null_aware(true),
        Arc::clone(&output_schema),
    );
    let plan = builder.finish_with_host_sink(join);

    let result = execute_query_plan(context, &plan).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    (0..batch.num_rows()).map(|row| int32_array(batch, 0).value(row)).collect()
}

#[test]
fn native_query_executes_null_aware_left_anti_join_not_in_semantics() {
    let dir = unique_temp_dir("query-engine-native-null-aware-anti-join");
    let context = backend_context();
    let runtime = Arc::new(context.runtime().unwrap());

    assert_eq!(
        null_aware_anti_join_output_qtys(
            &context,
            &runtime,
            &dir,
            "null-right-key",
            vec![Some(1), Some(2)],
            vec![Some(1), None]
        ),
        Vec::<i32>::new(),
        "a NULL right key should empty a null-aware anti join"
    );
    assert_eq!(
        null_aware_anti_join_output_qtys(&context, &runtime, &dir, "null-left-key", vec![Some(1), None], vec![Some(9)]),
        vec![0],
        "a NULL left key should drop out when the right input is non-empty"
    );

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_query_executes_left_mark_join_with_boolean_mark_column() {
    let dir = unique_temp_dir("query-engine-native-left-mark-join");
    let left_path = dir.join("lineitem.parquet");
    let right_path = dir.join("orders.parquet");
    let context = backend_context();
    let runtime = Arc::new(context.runtime().unwrap());
    let left_schema = write_join_left_parquet(&runtime, &left_path);
    let right_schema = write_join_right_parquet(&runtime, &right_path);
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("qty", DataType::Int32, false),
        Field::new("mark", DataType::Boolean, false),
    ]));

    let left_source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&left_path).with_file_size_bytes(fs::metadata(&left_path).unwrap().len())],
        Arc::clone(&left_schema),
        vec!["order_id".to_string(), "qty".to_string()],
    );
    let right_source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&right_path).with_file_size_bytes(fs::metadata(&right_path).unwrap().len())],
        Arc::clone(&right_schema),
        vec!["order_id".to_string()],
    );

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(left_source));
    let right = builder.add_source(SourceNode::LocalParquet(right_source));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(
            JoinType::LeftMark,
            vec![JoinKey::new(
                query_engine::expr::ColumnRef::input(0, "order_id", "left_order_id"),
                query_engine::expr::ColumnRef::input(1, "order_id", "right_order_id"),
            )],
        )
        .with_output_projection(vec![JoinOutputColumn::left(
            query_engine::expr::ColumnRef::input(0, "qty", "qty"),
            "qty",
        )])
        .with_null_equality(JoinNullEquality::NullsDoNotMatch)
        .with_build_side(JoinSide::Right)
        .with_mark_column_name("mark"),
        Arc::clone(&output_schema),
    );
    let sort = builder.add_sort(
        join,
        SortSpec::new(vec![SortExpr::new(Expr::column("qty"), false, false)]),
        Arc::clone(&output_schema),
    );
    let plan = builder.finish_with_host_sink(sort);

    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    let join_metrics =
        metrics.operators.iter().find_map(|operator| operator.join.as_ref()).expect("join metrics should be recorded");
    assert_eq!(join_metrics.join_type, "left_mark");
    assert_eq!(join_metrics.output_rows, 4);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    // Left data: order_id=[1,2,2,3] qty=[10,20,25,30]
    // Right data: order_id=[2,3,4]
    // Mark: order_id 1->false, 2->true, 2->true, 3->true
    // Sorted by qty asc: qty=[10,20,25,30], mark=[false,true,true,true]
    assert_eq!(batch.num_rows(), 4);
    assert_eq!(batch.num_columns(), 2);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[10, 20, 25, 30]);
    let mark = boolean_array(batch, 1);
    assert!(!mark.value(0));
    assert!(mark.value(1));
    assert!(mark.value(2));
    assert!(mark.value(3));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_query_executes_left_mark_join_with_no_matches() {
    let dir = unique_temp_dir("query-engine-native-left-mark-join-no-matches");
    let left_path = dir.join("lineitem.parquet");
    let right_path = dir.join("orders.parquet");
    let context = backend_context();
    let runtime = Arc::new(context.runtime().unwrap());
    let left_schema = write_join_left_parquet(&runtime, &left_path);
    // Write right side with no overlapping keys.
    let right_schema = write_join_right_keys_parquet(&runtime, &right_path, &[99, 100]);

    let output_schema = Arc::new(Schema::new(vec![
        Field::new("qty", DataType::Int32, false),
        Field::new("exists", DataType::Boolean, false),
    ]));

    let left_source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&left_path).with_file_size_bytes(fs::metadata(&left_path).unwrap().len())],
        Arc::clone(&left_schema),
        vec!["order_id".to_string(), "qty".to_string()],
    );
    let right_source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&right_path).with_file_size_bytes(fs::metadata(&right_path).unwrap().len())],
        Arc::clone(&right_schema),
        vec!["order_id".to_string()],
    );

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(left_source));
    let right = builder.add_source(SourceNode::LocalParquet(right_source));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(
            JoinType::LeftMark,
            vec![JoinKey::new(
                query_engine::expr::ColumnRef::input(0, "order_id", "left_order_id"),
                query_engine::expr::ColumnRef::input(1, "order_id", "right_order_id"),
            )],
        )
        .with_output_projection(vec![JoinOutputColumn::left(
            query_engine::expr::ColumnRef::input(0, "qty", "qty"),
            "qty",
        )])
        .with_null_equality(JoinNullEquality::NullsDoNotMatch)
        .with_build_side(JoinSide::Right)
        .with_mark_column_name("exists"),
        Arc::clone(&output_schema),
    );
    let sort = builder.add_sort(
        join,
        SortSpec::new(vec![SortExpr::new(Expr::column("qty"), false, false)]),
        Arc::clone(&output_schema),
    );
    let plan = builder.finish_with_host_sink(sort);

    let result = execute_query_plan(&context, &plan).unwrap();

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    // No matches: all marks false
    assert_eq!(batch.num_rows(), 4);
    assert_eq!(batch.num_columns(), 2);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[10, 20, 25, 30]);
    let mark = boolean_array(batch, 1);
    assert!(!mark.value(0));
    assert!(!mark.value(1));
    assert!(!mark.value(2));
    assert!(!mark.value(3));

    fs::remove_dir_all(dir).unwrap();
}
