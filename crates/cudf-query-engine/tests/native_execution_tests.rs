use std::{
    fs::{self, File},
    path::Path,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use arrow::{
    array::{
        Array, BooleanArray, Date32Array, Decimal128Array, Float64Array, Int32Array, Int64Array, StringArray,
        TimestampNanosecondArray,
    },
    datatypes::{DataType, Field, Schema, TimeUnit},
    record_batch::RecordBatch,
};
use cudf::DataFrame;
use cudf_query_engine::{
    AggregateExpr, AggregateFunction, AggregateSpec, BinaryOp, BoundedJoinReservationContract,
    BuildMaterializationBoundary, ExecutionBoundaryPolicy, ExecutionContext, Expr, GpuHashJoinBuildCache,
    GpuResidentInputMap, GpuResidentSource, JoinKey, JoinNullEquality, JoinOutputColumn, JoinSide, JoinSpec, JoinType,
    LimitSpec, LiteralValue, LocalParquetFile, LocalParquetSource, MemoryControlPolicy, PartitionColumn,
    QueryErrorCode, QueryMetrics, QueryOutput, QueryPlanBuilder, SortExpr, SortSpec, SourceChunkPlanningPolicy,
    SourceNode, ZERO_COLUMN_COUNT_COLUMN, execute_query_plan, execute_query_plan_with_gpu_inputs,
    materialize_gpu_result_to_host, plan_local_parquet_source_chunks,
};
use parquet::{arrow::ArrowWriter, file::properties::WriterProperties};

#[test]
fn native_query_executes_direct_parquet_pipeline_to_host_sink() {
    let dir = unique_temp_dir("cudf-query-engine-native-execution");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_input_parquet(&runtime, &path);

    let projected_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("x00", DataType::Int64, false),
    ]));
    let aggregated_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("sum_x00", DataType::Int64, true),
    ]));

    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["key".to_string(), "m00".to_string(), "m01".to_string(), "payload".to_string()],
    );
    let filter = Expr::binary(Expr::column("payload"), BinaryOp::GtEq, Expr::literal(LiteralValue::Int64(10)));
    let projection =
        vec![Expr::column("key"), Expr::binary(Expr::column("m00"), BinaryOp::Add, Expr::column("m01")).alias("x00")];
    let aggregate = AggregateSpec::new(
        vec![Expr::column("key")],
        vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("x00"), "sum_x00")],
    );
    let sort = SortSpec::new(vec![
        SortExpr::new(Expr::column("sum_x00"), true, false),
        SortExpr::new(Expr::column("key"), false, false),
    ]);

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let filter = builder.add_filter(source, filter, Arc::clone(&input_schema));
    let projection = builder.add_projection(filter, projection, projected_schema);
    let aggregate = builder.add_aggregate(projection, aggregate, Arc::clone(&aggregated_schema));
    let sort = builder.add_sort(aggregate, sort, Arc::clone(&aggregated_schema));
    let limit = builder.add_limit(sort, LimitSpec::fetch(2), aggregated_schema);
    let plan = builder.finish_with_host_sink(limit);

    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime));
    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.parquet_files, 1);
    assert_eq!(metrics.read_columns, 4);
    assert_eq!(metrics.input_batches, 1);
    assert_eq!(metrics.input_rows, 4);
    assert_eq!(metrics.boundary.host_to_device_uploads, 0);
    assert_eq!(metrics.boundary.device_to_host_materializations, 1);
    assert_eq!(metrics.boundary.device_to_host_materialized_rows, 2);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 2);
    assert_eq!(batch.num_columns(), 2);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[3, 1]);
    assert_eq!(int64_array(batch, 1).values().as_ref(), &[404, 303]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_execution_boundary_policy_reports_direct_source_and_explicit_host_materialization() {
    let dir = unique_temp_dir("cudf-query-engine-native-boundary-policy-source-host");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_input_parquet(&runtime, &path);
    let policy = ExecutionBoundaryPolicy::default();

    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["key".to_string(), "payload".to_string(), "m00".to_string(), "m01".to_string()],
    );

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let plan = builder.finish_with_host_sink(source);
    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_boundary_policy(policy)
        .build()
        .unwrap();

    assert_eq!(context.boundary_policy(), policy);
    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.execution_boundary_policy, policy);
    assert_eq!(metrics.boundary.host_to_device_uploads, 0);
    assert_eq!(metrics.boundary.device_to_host_materializations, 1);
    assert_eq!(metrics.boundary.device_to_host_materialized_rows, 4);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("explicit host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 4);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_query_executes_group_only_aggregate_as_distinct_keys() {
    let dir = unique_temp_dir("cudf-query-engine-native-group-only-aggregate");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
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

    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime));
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
fn native_query_executes_multi_file_direct_parquet_source() {
    let dir = unique_temp_dir("cudf-query-engine-native-execution-multi-file");
    let first_path = dir.join("part-000.parquet");
    let second_path = dir.join("part-001.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_input_parquet(&runtime, &first_path);
    write_input_parquet(&runtime, &second_path);

    let projected_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("x00", DataType::Int64, false),
    ]));
    let aggregated_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("sum_x00", DataType::Int64, true),
    ]));
    let files = vec![
        LocalParquetFile::new(&first_path).with_file_size_bytes(fs::metadata(&first_path).unwrap().len()),
        LocalParquetFile::new(&second_path).with_file_size_bytes(fs::metadata(&second_path).unwrap().len()),
    ];
    let source = LocalParquetSource::from_files(
        files,
        Arc::clone(&input_schema),
        vec!["key".to_string(), "m00".to_string(), "m01".to_string(), "payload".to_string()],
    );
    let filter = Expr::binary(Expr::column("payload"), BinaryOp::GtEq, Expr::literal(LiteralValue::Int64(10)));
    let projection =
        vec![Expr::column("key"), Expr::binary(Expr::column("m00"), BinaryOp::Add, Expr::column("m01")).alias("x00")];
    let aggregate = AggregateSpec::new(
        vec![Expr::column("key")],
        vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("x00"), "sum_x00")],
    );
    let sort = SortSpec::new(vec![
        SortExpr::new(Expr::column("sum_x00"), true, false),
        SortExpr::new(Expr::column("key"), false, false),
    ]);

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let filter = builder.add_filter(source, filter, Arc::clone(&input_schema));
    let projection = builder.add_projection(filter, projection, projected_schema);
    let aggregate = builder.add_aggregate(projection, aggregate, Arc::clone(&aggregated_schema));
    let sort = builder.add_sort(aggregate, sort, Arc::clone(&aggregated_schema));
    let limit = builder.add_limit(sort, LimitSpec::fetch(2), aggregated_schema);
    let plan = builder.finish_with_host_sink(limit);

    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime));
    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.parquet_files, 2);
    assert_eq!(metrics.read_columns, 8);
    assert_eq!(metrics.input_batches, 2);
    assert_eq!(metrics.input_rows, 8);
    assert_eq!(metrics.boundary.host_to_device_uploads, 0);
    assert_eq!(metrics.boundary.device_to_host_materializations, 1);
    assert_eq!(metrics.boundary.device_to_host_materialized_rows, 2);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[3, 1]);
    assert_eq!(int64_array(batch, 1).values().as_ref(), &[808, 606]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_query_executes_inner_equi_join_with_explicit_output_projection() {
    let dir = unique_temp_dir("cudf-query-engine-native-inner-join");
    let left_path = dir.join("lineitem.parquet");
    let right_path = dir.join("orders.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
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
                cudf_query_engine::ColumnRef::input(0, "order_id", "left_order_id"),
                cudf_query_engine::ColumnRef::input(1, "order_id", "right_order_id"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(cudf_query_engine::ColumnRef::input(0, "order_id", "order_id"), "order_id"),
            JoinOutputColumn::left(cudf_query_engine::ColumnRef::input(0, "qty", "qty"), "qty"),
            JoinOutputColumn::right(cudf_query_engine::ColumnRef::input(1, "category", "category"), "category"),
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

    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime));
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
    let dir = unique_temp_dir("cudf-query-engine-native-inner-join-without-key-output");
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
                cudf_query_engine::ColumnRef::input(0, "order_id", "left_order_id"),
                cudf_query_engine::ColumnRef::input(1, "order_id", "right_order_id"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(cudf_query_engine::ColumnRef::input(0, "qty", "qty"), "qty"),
            JoinOutputColumn::right(cudf_query_engine::ColumnRef::input(1, "category", "category"), "category"),
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

    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime));
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
    let dir = unique_temp_dir("cudf-query-engine-native-left-join");
    let left_path = dir.join("lineitem.parquet");
    let right_path = dir.join("orders.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
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
                cudf_query_engine::ColumnRef::input(0, "order_id", "left_order_id"),
                cudf_query_engine::ColumnRef::input(1, "order_id", "right_order_id"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(cudf_query_engine::ColumnRef::input(0, "qty", "qty"), "qty"),
            JoinOutputColumn::right(cudf_query_engine::ColumnRef::input(1, "category", "category"), "category"),
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

    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime));
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
    let dir = unique_temp_dir("cudf-query-engine-native-full-join");
    let left_path = dir.join("lineitem.parquet");
    let right_path = dir.join("orders.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
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
                cudf_query_engine::ColumnRef::input(0, "order_id", "left_order_id"),
                cudf_query_engine::ColumnRef::input(1, "order_id", "right_order_id"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(
                cudf_query_engine::ColumnRef::input(0, "order_id", "left_order_id"),
                "left_order_id",
            ),
            JoinOutputColumn::right(
                cudf_query_engine::ColumnRef::input(1, "order_id", "right_order_id"),
                "right_order_id",
            ),
            JoinOutputColumn::left(cudf_query_engine::ColumnRef::input(0, "qty", "qty"), "qty"),
            JoinOutputColumn::right(cudf_query_engine::ColumnRef::input(1, "category", "category"), "category"),
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

    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime));
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
    let dir = unique_temp_dir("cudf-query-engine-native-left-semi-join");
    let left_path = dir.join("lineitem.parquet");
    let right_path = dir.join("orders.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
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
                cudf_query_engine::ColumnRef::input(0, "order_id", "left_order_id"),
                cudf_query_engine::ColumnRef::input(1, "order_id", "right_order_id"),
            )],
        )
        .with_output_projection(vec![JoinOutputColumn::left(
            cudf_query_engine::ColumnRef::input(0, "qty", "qty"),
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

    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime));
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
    let dir = unique_temp_dir("cudf-query-engine-native-left-anti-join");
    let left_path = dir.join("lineitem.parquet");
    let right_path = dir.join("orders.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
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
                cudf_query_engine::ColumnRef::input(0, "order_id", "left_order_id"),
                cudf_query_engine::ColumnRef::input(1, "order_id", "right_order_id"),
            )],
        )
        .with_output_projection(vec![JoinOutputColumn::left(
            cudf_query_engine::ColumnRef::input(0, "qty", "qty"),
            "qty",
        )])
        .with_null_equality(JoinNullEquality::NullsDoNotMatch)
        .with_build_side(JoinSide::Right),
        Arc::clone(&output_schema),
    );
    let plan = builder.finish_with_host_sink(join);

    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime));
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

#[test]
fn native_query_executes_cross_join_with_explicit_output_projection() {
    let dir = unique_temp_dir("cudf-query-engine-native-cross-join");
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
            JoinOutputColumn::left(cudf_query_engine::ColumnRef::input(0, "qty", "qty"), "qty"),
            JoinOutputColumn::right(cudf_query_engine::ColumnRef::input(1, "category", "category"), "category"),
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

    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime));
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
fn native_query_executes_inner_join_with_equi_key_and_residual_predicate() {
    let dir = unique_temp_dir("cudf-query-engine-native-mixed-join");
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
        vec!["order_id".to_string(), "qty".to_string()],
    );
    let right_source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&right_path).with_file_size_bytes(fs::metadata(&right_path).unwrap().len())],
        Arc::clone(&right_schema),
        vec!["order_id".to_string(), "category".to_string()],
    );
    let residual = Expr::binary(
        Expr::input_column(0, "qty", "qty"),
        BinaryOp::Gt,
        Expr::binary(
            Expr::input_column(1, "order_id", "right_order_id"),
            BinaryOp::Multiply,
            Expr::literal(LiteralValue::Int32(10)),
        ),
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
                cudf_query_engine::ColumnRef::input(0, "order_id", "left_order_id"),
                cudf_query_engine::ColumnRef::input(1, "order_id", "right_order_id"),
            )],
        )
        .with_residual_predicate(residual)
        .with_output_projection(vec![
            JoinOutputColumn::left(cudf_query_engine::ColumnRef::input(0, "qty", "qty"), "qty"),
            JoinOutputColumn::right(cudf_query_engine::ColumnRef::input(1, "category", "category"), "category"),
        ])
        .with_null_equality(JoinNullEquality::NullsDoNotMatch)
        .with_build_side(JoinSide::Right),
        Arc::clone(&output_schema),
    );
    let plan = builder.finish_with_host_sink(join);

    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime));
    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.hash_join_build_count, 1);
    assert_eq!(metrics.hash_join_cache_misses, 1);
    let join_metrics =
        metrics.operators.iter().find_map(|operator| operator.join.as_ref()).expect("join metrics should be recorded");
    assert!(join_metrics.residual_predicate);
    assert_eq!(join_metrics.output_rows, 1);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 1);
    assert_eq!(int32_array(batch, 0).value(0), 25);
    assert_eq!(string_array(batch, 1).value(0), "two");

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_query_executes_join_residual_cast_predicate() {
    let dir = unique_temp_dir("cudf-query-engine-native-mixed-join-cast");
    let left_path = dir.join("lineitem.parquet");
    let right_path = dir.join("orders.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
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
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(JoinType::Inner, vec![join_order_id_key()])
            .with_residual_predicate(residual)
            .with_output_projection(vec![
                JoinOutputColumn::left(cudf_query_engine::ColumnRef::input(0, "qty", "qty"), "qty"),
                JoinOutputColumn::right(cudf_query_engine::ColumnRef::input(1, "category", "category"), "category"),
            ])
            .with_build_side(JoinSide::Right),
        Arc::clone(&output_schema),
    );
    let plan = builder.finish_with_host_sink(join);

    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime));
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
    let dir = unique_temp_dir("cudf-query-engine-native-mixed-join-decimal-cast");
    let left_path = dir.join("lineitem.parquet");
    let right_path = dir.join("orders.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
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
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(JoinType::Inner, vec![join_order_id_key()])
            .with_residual_predicate(residual)
            .with_output_projection(vec![
                JoinOutputColumn::left(cudf_query_engine::ColumnRef::input(0, "amount", "amount"), "amount"),
                JoinOutputColumn::right(cudf_query_engine::ColumnRef::input(1, "category", "category"), "category"),
            ])
            .with_build_side(JoinSide::Right),
        Arc::clone(&output_schema),
    );
    let plan = builder.finish_with_host_sink(join);

    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime));
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
    let dir = unique_temp_dir("cudf-query-engine-native-join-residual-in-list");
    let left_path = dir.join("lineitem.parquet");
    let right_path = dir.join("orders.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
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
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(JoinType::Inner, vec![join_order_id_key()])
            .with_residual_predicate(residual)
            .with_output_projection(vec![
                JoinOutputColumn::left(cudf_query_engine::ColumnRef::input(0, "qty", "qty"), "qty"),
                JoinOutputColumn::right(cudf_query_engine::ColumnRef::input(1, "category", "category"), "category"),
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

    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime));
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

#[test]
fn native_query_reuses_build_side_hash_join_for_repeated_broadcast_probe() {
    let dir = unique_temp_dir("cudf-query-engine-native-hash-join-cache");
    let left_path = dir.join("lineitem.parquet");
    let right_path = dir.join("orders.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let left_schema = write_join_left_parquet(&runtime, &left_path);
    let right_schema = write_join_right_parquet(&runtime, &right_path);
    let first_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
    ]));
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
    let first = builder.add_join(
        left,
        right,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                cudf_query_engine::ColumnRef::input(0, "order_id", "left_order_id"),
                cudf_query_engine::ColumnRef::input(1, "order_id", "right_order_id"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(cudf_query_engine::ColumnRef::input(0, "order_id", "order_id"), "order_id"),
            JoinOutputColumn::left(cudf_query_engine::ColumnRef::input(0, "qty", "qty"), "qty"),
        ])
        .with_build_side(JoinSide::Right),
        Arc::clone(&first_schema),
    );
    let second = builder.add_join(
        first,
        right,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                cudf_query_engine::ColumnRef::input(0, "order_id", "left_order_id"),
                cudf_query_engine::ColumnRef::input(1, "order_id", "right_order_id"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(cudf_query_engine::ColumnRef::input(0, "qty", "qty"), "qty"),
            JoinOutputColumn::right(cudf_query_engine::ColumnRef::input(1, "category", "category"), "category"),
        ])
        .with_build_side(JoinSide::Right),
        Arc::clone(&output_schema),
    );
    let sort = builder.add_sort(
        second,
        SortSpec::new(vec![SortExpr::new(Expr::column("qty"), false, false)]),
        Arc::clone(&output_schema),
    );
    let plan = builder.finish_with_host_sink(sort);
    let policy = ExecutionBoundaryPolicy::default();

    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_boundary_policy(policy)
        .build()
        .unwrap();
    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.execution_boundary_policy, policy);
    assert_eq!(metrics.parquet_files, 2);
    assert_eq!(metrics.hash_join_build_count, 1);
    assert_eq!(metrics.hash_join_cache_misses, 1);
    assert_eq!(metrics.hash_join_cache_hits, 1);
    assert!(metrics.hash_join_hash_table_bytes > 0);
    let join_metrics = metrics.operators.iter().filter_map(|operator| operator.join.as_ref()).collect::<Vec<_>>();
    let join_count = join_metrics.len();
    assert_eq!(join_count, 2);
    assert!(join_metrics.iter().all(|join| join.hash_table_bytes == metrics.hash_join_hash_table_bytes));
    assert!(join_metrics.iter().all(|join| join.gather_map_bytes > 0));

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 3);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[20, 25, 30]);
    assert_eq!(string_array(batch, 1).iter().collect::<Vec<_>>(), vec![Some("two"), Some("two"), Some("three")]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_query_reuses_durable_hash_join_build_across_queries() {
    let dir = unique_temp_dir("cudf-query-engine-native-durable-hash-join-cache");
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
        JoinSpec::new(JoinType::Inner, vec![join_order_id_key()])
            .with_output_projection(vec![
                JoinOutputColumn::left(cudf_query_engine::ColumnRef::input(0, "qty", "qty"), "qty"),
                JoinOutputColumn::right(cudf_query_engine::ColumnRef::input(1, "category", "category"), "category"),
            ])
            .with_build_side(JoinSide::Right)
            .with_build_artifact_id("orders_by_order_id"),
        Arc::clone(&output_schema),
    );
    let sort = builder.add_sort(
        join,
        SortSpec::new(vec![SortExpr::new(Expr::column("qty"), false, false)]),
        Arc::clone(&output_schema),
    );
    let plan = builder.finish_with_host_sink(sort);
    let cache = Arc::new(GpuHashJoinBuildCache::new());
    let policy =
        ExecutionBoundaryPolicy::default().with_build_materialization(BuildMaterializationBoundary::DurableGpuHashJoin);
    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_boundary_policy(policy)
        .with_hash_join_build_cache(Arc::clone(&cache))
        .build()
        .unwrap();

    let first = execute_query_plan(&context, &plan).unwrap();
    assert_eq!(first.metrics().hash_join_build_count, 1);
    assert_eq!(first.metrics().hash_join_cache_misses, 1);
    assert_eq!(first.metrics().hash_join_durable_cache_misses, 1);
    assert_eq!(first.metrics().hash_join_durable_cache_inserts, 1);
    assert_eq!(first.metrics().hash_join_durable_cache_hits, 0);
    assert_eq!(cache.len().unwrap(), 1);
    let retained_bytes = cache.retained_hash_table_bytes().unwrap();
    assert!(retained_bytes > 0);

    let second = execute_query_plan(&context, &plan).unwrap();
    assert_eq!(second.metrics().hash_join_build_count, 0);
    assert_eq!(second.metrics().hash_join_cache_misses, 1);
    assert_eq!(second.metrics().hash_join_durable_cache_hits, 1);
    assert_eq!(second.metrics().hash_join_durable_cache_misses, 0);
    assert_eq!(second.metrics().hash_join_durable_cache_inserts, 0);
    assert_eq!(second.metrics().hash_join_hash_table_bytes, retained_bytes);

    let QueryOutput::HostArrow(batch) = second.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 3);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[20, 25, 30]);
    assert_eq!(string_array(batch, 1).iter().collect::<Vec<_>>(), vec![Some("two"), Some("two"), Some("three")]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_query_executes_conditional_non_equi_inner_join() {
    let dir = unique_temp_dir("cudf-query-engine-native-conditional-inner-join");
    let left_path = dir.join("lineitem.parquet");
    let right_path = dir.join("orders.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
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
                JoinOutputColumn::left(cudf_query_engine::ColumnRef::input(0, "qty", "qty"), "qty"),
                JoinOutputColumn::right(
                    cudf_query_engine::ColumnRef::input(1, "order_id", "right_order_id"),
                    "right_order_id",
                ),
                JoinOutputColumn::right(cudf_query_engine::ColumnRef::input(1, "category", "category"), "category"),
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

    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime));
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
    let dir = unique_temp_dir("cudf-query-engine-native-mixed-left-join");
    let left_path = dir.join("lineitem.parquet");
    let right_path = dir.join("orders.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
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
                JoinOutputColumn::left(cudf_query_engine::ColumnRef::input(0, "qty", "qty"), "qty"),
                JoinOutputColumn::right(cudf_query_engine::ColumnRef::input(1, "category", "category"), "category"),
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

    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime));
    let result = execute_query_plan(&context, &plan).unwrap();
    let join_metrics = result
        .metrics()
        .operators
        .iter()
        .find_map(|operator| operator.join.as_ref())
        .expect("join metrics should be recorded");
    assert_eq!(join_metrics.join_type, "left");
    assert_eq!(join_metrics.key_count, 1);
    assert!(join_metrics.residual_predicate);
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
    let dir = unique_temp_dir("cudf-query-engine-native-mixed-full-join");
    let left_path = dir.join("lineitem.parquet");
    let right_path = dir.join("orders.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
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
                    cudf_query_engine::ColumnRef::input(0, "order_id", "left_order_id"),
                    "left_order_id",
                ),
                JoinOutputColumn::right(
                    cudf_query_engine::ColumnRef::input(1, "order_id", "right_order_id"),
                    "right_order_id",
                ),
                JoinOutputColumn::left(cudf_query_engine::ColumnRef::input(0, "qty", "qty"), "qty"),
                JoinOutputColumn::right(cudf_query_engine::ColumnRef::input(1, "category", "category"), "category"),
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

    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime));
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
    let dir = unique_temp_dir("cudf-query-engine-native-mixed-semi-anti-join");
    let left_path = dir.join("lineitem.parquet");
    let right_path = dir.join("orders.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
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

#[test]
fn native_query_rejects_invalid_cross_join_before_source_read() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let left_schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    let right_schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("left_id", DataType::Int32, false),
        Field::new("right_id", DataType::Int32, true),
    ]));
    let left_source = LocalParquetSource::new(
        "/tmp/does-not-need-to-exist-for-cross-join-left.parquet",
        Arc::clone(&left_schema),
        vec!["id".into()],
    );
    let right_source = LocalParquetSource::new(
        "/tmp/does-not-need-to-exist-for-cross-join-right.parquet",
        Arc::clone(&right_schema),
        vec!["id".into()],
    );

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(left_source));
    let right = builder.add_source(SourceNode::LocalParquet(right_source));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(
            JoinType::Cross,
            vec![JoinKey::new(
                cudf_query_engine::ColumnRef::input(0, "id", "left_id"),
                cudf_query_engine::ColumnRef::input(1, "id", "right_id"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(cudf_query_engine::ColumnRef::input(0, "id", "left_id"), "left_id"),
            JoinOutputColumn::right(cudf_query_engine::ColumnRef::input(1, "id", "right_id"), "right_id"),
        ]),
        output_schema,
    );
    let plan = builder.finish_with_host_sink(join);
    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime));

    let error = match execute_query_plan(&context, &plan) {
        Ok(_) => panic!("native validation should reject cross joins with equality keys before source read"),
        Err(error) => error,
    };
    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.operation(), "cudf_query_engine.join.validate");
    assert_eq!(error.fact_value("key_count"), Some("1"));
    assert!(error.fact_value("path").is_none());
}

#[test]
fn native_query_rejects_cross_join_with_residual_before_source_read() {
    let error = residual_validation_error(
        JoinType::Cross,
        vec![],
        Expr::binary(Expr::input_column(0, "id", "left_id"), BinaryOp::Eq, Expr::input_column(1, "id", "right_id")),
    );

    assert_eq!(error.code(), QueryErrorCode::UnsupportedPlanShape);
    assert_eq!(error.operation(), "cudf_query_engine.join.validate");
    assert_eq!(error.fact_value("join_type"), Some("cross"));
    assert!(error.fact_value("path").is_none());
}

#[test]
fn native_query_rejects_unqualified_residual_column_before_source_read() {
    let error = residual_validation_error(
        JoinType::Inner,
        vec![id_join_key()],
        Expr::binary(Expr::column("id"), BinaryOp::Eq, Expr::input_column(1, "id", "right_id")),
    );

    assert_eq!(error.code(), QueryErrorCode::UnsupportedExpression);
    assert_eq!(error.operation(), "cudf_query_engine.join.validate_residual");
    assert_eq!(error.fact_value("column"), Some("id"));
    assert_eq!(error.fact_value("qualifier"), Some("none"));
    assert!(error.fact_value("path").is_none());
}

#[test]
fn native_query_rejects_one_sided_residual_predicate_before_source_read() {
    let error = residual_validation_error(
        JoinType::Inner,
        vec![id_join_key()],
        Expr::binary(Expr::input_column(0, "id", "left_id"), BinaryOp::Gt, Expr::literal(LiteralValue::Int32(1))),
    );

    assert_eq!(error.code(), QueryErrorCode::UnsupportedExpression);
    assert_eq!(error.operation(), "cudf_query_engine.join.validate_residual");
    assert_eq!(error.fact_value("references_left"), Some("true"));
    assert_eq!(error.fact_value("references_right"), Some("false"));
    assert!(error.fact_value("path").is_none());
}

#[test]
fn native_query_rejects_non_boolean_residual_predicate_before_source_read() {
    let error = residual_validation_error(
        JoinType::Inner,
        vec![id_join_key()],
        Expr::binary(Expr::input_column(0, "id", "left_id"), BinaryOp::Add, Expr::input_column(1, "id", "right_id")),
    );

    assert_eq!(error.code(), QueryErrorCode::UnsupportedType);
    assert_eq!(error.operation(), "cudf_query_engine.plan.validate_expr");
    assert_eq!(error.fact_value("context"), Some("join residual predicate"));
    assert!(error.fact_value("path").is_none());
}

#[test]
fn native_query_rejects_join_residual_in_list_with_non_literal_values_before_source_read() {
    let error = residual_validation_error(
        JoinType::Inner,
        vec![id_join_key()],
        Expr::binary(
            Expr::in_list(Expr::input_column(0, "id", "left_id"), vec![Expr::input_column(1, "id", "right_id")]),
            BinaryOp::And,
            Expr::binary(Expr::input_column(0, "id", "left_id"), BinaryOp::Eq, Expr::input_column(1, "id", "right_id")),
        ),
    );

    assert_eq!(error.code(), QueryErrorCode::UnsupportedExpression);
    assert_eq!(error.operation(), "cudf_query_engine.join.validate_residual");
    assert!(error.message().contains("IN list values must be literals"), "unexpected error: {error}");
    assert!(error.fact_value("path").is_none());
}

#[test]
fn native_query_rejects_unsupported_join_residual_cast_before_source_read() {
    let error = residual_validation_error(
        JoinType::Inner,
        vec![id_join_key()],
        Expr::binary(
            Expr::cast(Expr::input_column(0, "id", "left_id"), DataType::Int32),
            BinaryOp::Eq,
            Expr::input_column(1, "id", "right_id"),
        ),
    );

    assert_eq!(error.code(), QueryErrorCode::UnsupportedExpression);
    assert_eq!(error.operation(), "cudf_query_engine.join.validate_residual");
    assert_eq!(error.fact_value("target_type"), Some("Int32"));
    assert!(error.fact_value("path").is_none());
}

#[test]
fn native_query_rejects_unsupported_residual_expression_families_before_source_read() {
    let cases = vec![
        ("LIKE", Expr::like(Expr::input_column(0, "id", "left_id"), "1%")),
        ("SUBSTRING", Expr::substring(Expr::input_column(0, "id", "left_id"), 1, Some(1))),
        (
            "CASE",
            Expr::case(
                vec![cudf_query_engine::CaseWhen::new(
                    Expr::binary(
                        Expr::input_column(0, "id", "left_id"),
                        BinaryOp::Eq,
                        Expr::input_column(1, "id", "right_id"),
                    ),
                    Expr::literal(LiteralValue::Boolean(true)),
                )],
                Expr::literal(LiteralValue::Boolean(false)),
            ),
        ),
        ("EXTRACT", Expr::extract(cudf_query_engine::DatePart::Year, Expr::input_column(0, "id", "left_id"))),
    ];

    for (family, residual) in cases {
        let error = residual_validation_error(JoinType::Inner, vec![id_join_key()], residual);
        assert_eq!(error.code(), QueryErrorCode::UnsupportedExpression, "{family}");
        assert_eq!(error.operation(), "cudf_query_engine.join.validate_residual", "{family}");
        assert!(
            error.fact_value("expression").is_some_and(|expression| expression.contains(family)),
            "{family} should be visible in expression fact: {:?}",
            error.fact_value("expression")
        );
        assert!(error.fact_value("path").is_none(), "{family}");
    }
}

#[test]
fn native_gpu_result_handle_rejects_sink_schema_name_order_mismatch_before_host_materialization() {
    let dir = unique_temp_dir("cudf-query-engine-native-execution-schema-mismatch");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_input_parquet(&runtime, &path);
    let mismatched_schema = Arc::new(Schema::new(vec![
        Field::new("payload", DataType::Int64, false),
        Field::new("key", DataType::Int32, false),
        Field::new("m00", DataType::Int64, false),
        Field::new("m01", DataType::Int64, false),
    ]));
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["key".to_string(), "payload".to_string(), "m00".to_string(), "m01".to_string()],
    );

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let limit = builder.add_limit(source, LimitSpec::fetch(2), mismatched_schema);
    let plan = builder.finish_with_host_sink(limit);

    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime));
    let error = match execute_query_plan(&context, &plan) {
        Ok(_) => panic!("GPU result handle should reject sink schema name/order mismatches"),
        Err(error) => error,
    };
    assert_eq!(error.code(), QueryErrorCode::CudfInvalidInput);
    assert_eq!(error.operation(), "cudf_query_engine.gpu_result_handle");
    assert_eq!(error.fact_value("cudf_operation"), Some("dataframe::GpuRecordBatch::new"));
    assert_eq!(error.fact_value("cudf_status"), Some("permanent"));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_query_executes_filter_projection_over_tpch_like_types() {
    let dir = unique_temp_dir("cudf-query-engine-native-execution-tpch-like-types");
    let path = dir.join("tpch-like.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_tpch_like_types_parquet(&runtime, &path);
    let projected_schema = Arc::new(Schema::new(vec![
        Field::new("orderkey", DataType::Int64, false),
        Field::new("amount", DataType::Decimal128(15, 2), false),
        Field::new("shipdate", DataType::Date32, false),
        Field::new("shipdate_plus_month", DataType::Date32, false),
        Field::new("event_ts", DataType::Timestamp(TimeUnit::Nanosecond, None), false),
        Field::new("event_ts_plus", DataType::Timestamp(TimeUnit::Nanosecond, None), false),
        Field::new("event_ts_plus_interval", DataType::Timestamp(TimeUnit::Nanosecond, None), false),
        Field::new("status", DataType::Utf8, false),
        Field::new("large_literal", DataType::Utf8, false),
        Field::new("null_amount", DataType::Decimal128(15, 2), true),
        Field::new("null_shipdate", DataType::Date32, true),
        Field::new("null_event_ts", DataType::Timestamp(TimeUnit::Nanosecond, None), true),
        Field::new("null_status", DataType::Utf8, true),
    ]));
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec![
            "orderkey".to_string(),
            "amount".to_string(),
            "shipdate".to_string(),
            "event_ts".to_string(),
            "status".to_string(),
        ],
    );
    let filter = Expr::binary(
        Expr::binary(
            Expr::binary(
                Expr::binary(
                    Expr::column("amount"),
                    BinaryOp::Gt,
                    Expr::literal(LiteralValue::Decimal128 { raw: 20_000, precision: 15, scale: 2 }),
                ),
                BinaryOp::And,
                Expr::binary(Expr::column("shipdate"), BinaryOp::GtEq, Expr::literal(LiteralValue::Date32(18_322))),
            ),
            BinaryOp::And,
            Expr::binary(
                Expr::column("event_ts"),
                BinaryOp::Lt,
                Expr::literal(LiteralValue::Timestamp { raw: 400, unit: TimeUnit::Nanosecond }),
            ),
        ),
        BinaryOp::And,
        Expr::binary(Expr::column("status"), BinaryOp::Eq, Expr::literal(LiteralValue::Utf8("O".to_string()))),
    );
    let projection = vec![
        Expr::column("orderkey"),
        Expr::column("amount"),
        Expr::column("shipdate"),
        Expr::binary(
            Expr::column("shipdate"),
            BinaryOp::Add,
            Expr::literal(LiteralValue::IntervalMonthDayNano { months: 1, days: 0, nanoseconds: 0 }),
        )
        .alias("shipdate_plus_month"),
        Expr::column("event_ts"),
        Expr::binary(
            Expr::column("event_ts"),
            BinaryOp::Add,
            Expr::literal(LiteralValue::Duration { raw: 10, unit: TimeUnit::Nanosecond }),
        )
        .alias("event_ts_plus"),
        Expr::binary(
            Expr::column("event_ts"),
            BinaryOp::Add,
            Expr::literal(LiteralValue::IntervalMonthDayNano { months: 0, days: 1, nanoseconds: 5 }),
        )
        .alias("event_ts_plus_interval"),
        Expr::column("status"),
        Expr::literal(LiteralValue::LargeUtf8("wide".to_string())).alias("large_literal"),
        Expr::literal(LiteralValue::Null(DataType::Decimal128(15, 2))).alias("null_amount"),
        Expr::literal(LiteralValue::Null(DataType::Date32)).alias("null_shipdate"),
        Expr::literal(LiteralValue::Null(DataType::Timestamp(TimeUnit::Nanosecond, None))).alias("null_event_ts"),
        Expr::literal(LiteralValue::Null(DataType::Utf8)).alias("null_status"),
    ];

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let filter = builder.add_filter(source, filter, Arc::clone(&input_schema));
    let projection = builder.add_projection(filter, projection, projected_schema);
    let plan = builder.finish_with_host_sink(projection);

    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime));
    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.parquet_files, 1);
    assert_eq!(metrics.read_columns, 5);
    assert_eq!(metrics.boundary.host_to_device_uploads, 0);
    assert_eq!(metrics.boundary.device_to_host_materializations, 1);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 1);
    assert_eq!(int64_array(batch, 0).value(0), 3);
    let amount = decimal128_array(batch, 1);
    assert_eq!(amount.value(0), 22_222);
    assert!(matches!(amount.data_type(), DataType::Decimal128(_, 2)));
    assert_eq!(date32_array(batch, 2).value(0), 18_322);
    assert_eq!(date32_array(batch, 3).value(0), 18_353);
    assert_eq!(timestamp_ns_array(batch, 4).value(0), 300);
    assert_eq!(timestamp_ns_array(batch, 5).value(0), 310);
    assert_eq!(timestamp_ns_array(batch, 6).value(0), 86_400_000_000_305);
    assert_eq!(string_array(batch, 7).value(0), "O");
    assert_eq!(string_array(batch, 8).value(0), "wide");
    for index in 9..13 {
        assert!(batch.column(index).is_null(0), "projection column {index} should be a typed null");
    }

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_query_executes_phase2_scalar_expression_families() {
    let dir = unique_temp_dir("cudf-query-engine-native-execution-phase2-scalars");
    let path = dir.join("tpch-like.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_phase2_scalar_expression_parquet(&runtime, &path);
    let projected_schema = Arc::new(Schema::new(vec![
        Field::new("orderkey", DataType::Int64, false),
        Field::new("order_mod", DataType::Int64, false),
        Field::new("ship_year", DataType::Int32, false),
        Field::new("ship_month", DataType::Int32, false),
        Field::new("ship_day", DataType::Int32, false),
        Field::new("event_ts_minus_interval", DataType::Timestamp(TimeUnit::Nanosecond, None), false),
        Field::new("status_prefix", DataType::Utf8, true),
        Field::new("status_label", DataType::Utf8, false),
        Field::new("status_known", DataType::Boolean, false),
        Field::new("status_is_null", DataType::Boolean, false),
        Field::new("status_not_f", DataType::Boolean, true),
        Field::new("status_like_o", DataType::Boolean, true),
        Field::new("status_not_like_f", DataType::Boolean, true),
    ]));
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["orderkey".to_string(), "shipdate".to_string(), "event_ts".to_string(), "status".to_string()],
    );
    let filter = Expr::binary(
        Expr::in_list(
            Expr::column("orderkey"),
            vec![
                Expr::literal(LiteralValue::Int64(1)),
                Expr::literal(LiteralValue::Int64(3)),
                Expr::literal(LiteralValue::Int64(4)),
            ],
        ),
        BinaryOp::And,
        Expr::is_not_null(Expr::column("shipdate")),
    );
    let projection = vec![
        Expr::column("orderkey"),
        Expr::binary(Expr::column("orderkey"), BinaryOp::Modulo, Expr::literal(LiteralValue::Int64(2)))
            .alias("order_mod"),
        Expr::extract(cudf_query_engine::DatePart::Year, Expr::column("shipdate")).alias("ship_year"),
        Expr::extract(cudf_query_engine::DatePart::Month, Expr::column("shipdate")).alias("ship_month"),
        Expr::extract(cudf_query_engine::DatePart::Day, Expr::column("shipdate")).alias("ship_day"),
        Expr::binary(
            Expr::column("event_ts"),
            BinaryOp::Subtract,
            Expr::literal(LiteralValue::IntervalMonthDayNano { months: 0, days: 0, nanoseconds: 50 }),
        )
        .alias("event_ts_minus_interval"),
        Expr::substring(Expr::column("status"), 1, Some(1)).alias("status_prefix"),
        Expr::case(
            vec![cudf_query_engine::CaseWhen::new(
                Expr::binary(Expr::column("status"), BinaryOp::Eq, Expr::literal(LiteralValue::Utf8("O".to_string()))),
                Expr::literal(LiteralValue::Utf8("open".to_string())),
            )],
            Expr::literal(LiteralValue::Utf8("other".to_string())),
        )
        .alias("status_label"),
        Expr::is_not_null(Expr::column("status")).alias("status_known"),
        Expr::is_null(Expr::column("status")).alias("status_is_null"),
        Expr::not_in_list(Expr::column("status"), vec![Expr::literal(LiteralValue::Utf8("F".to_string()))])
            .alias("status_not_f"),
        Expr::like(Expr::column("status"), "O%").alias("status_like_o"),
        Expr::not_like(Expr::column("status"), "F%").alias("status_not_like_f"),
    ];

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let filter = builder.add_filter(source, filter, Arc::clone(&input_schema));
    let projection = builder.add_projection(filter, projection, projected_schema);
    let plan = builder.finish_with_host_sink(projection);

    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime));
    let result = execute_query_plan(&context, &plan).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 3);
    assert_eq!(int64_array(batch, 0).values().as_ref(), &[1, 3, 4]);
    assert_eq!(int64_array(batch, 1).values().as_ref(), &[1, 1, 0]);
    assert_eq!(int32_array(batch, 2).values().as_ref(), &[2020, 2020, 2020]);
    assert_eq!(int32_array(batch, 3).values().as_ref(), &[1, 3, 3]);
    assert_eq!(int32_array(batch, 4).values().as_ref(), &[1, 1, 15]);
    assert_eq!(timestamp_ns_array(batch, 5).values().as_ref(), &[50, 250, 350]);
    assert_eq!(string_array(batch, 6).value(0), "O");
    assert_eq!(string_array(batch, 6).value(1), "O");
    assert!(string_array(batch, 6).is_null(2));
    assert_eq!(string_array(batch, 7).value(0), "open");
    assert_eq!(string_array(batch, 7).value(1), "open");
    assert_eq!(string_array(batch, 7).value(2), "other");
    assert_eq!(boolean_array(batch, 8).values().iter().collect::<Vec<_>>(), vec![true, true, false]);
    assert_eq!(boolean_array(batch, 9).values().iter().collect::<Vec<_>>(), vec![false, false, true]);
    assert!(boolean_array(batch, 10).value(0));
    assert!(boolean_array(batch, 10).value(1));
    assert!(boolean_array(batch, 10).is_null(2));
    assert!(boolean_array(batch, 11).value(0));
    assert!(boolean_array(batch, 11).value(1));
    assert!(boolean_array(batch, 11).is_null(2));
    assert!(boolean_array(batch, 12).value(0));
    assert!(boolean_array(batch, 12).value(1));
    assert!(boolean_array(batch, 12).is_null(2));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_query_rejects_phase2_scalar_type_mismatch_before_cudf_execution() {
    let dir = unique_temp_dir("cudf-query-engine-native-execution-phase2-type-reject");
    let path = dir.join("tpch-like.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_phase2_scalar_expression_parquet(&runtime, &path);
    let output_schema = Arc::new(Schema::new(vec![Field::new("bad_like", DataType::Boolean, true)]));
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["orderkey".to_string()],
    );

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let projection = builder.add_projection(
        source,
        vec![Expr::like(Expr::column("orderkey"), "1%").alias("bad_like")],
        output_schema,
    );
    let plan = builder.finish_with_host_sink(projection);
    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime));

    let error = match execute_query_plan(&context, &plan) {
        Ok(_) => panic!("native plan validation should reject LIKE over an integral input before cuDF execution"),
        Err(error) => error,
    };
    assert_eq!(error.code(), QueryErrorCode::UnsupportedType);
    assert_eq!(error.operation(), "cudf_query_engine.plan.validate_expr");
    assert_eq!(error.fact_value("context"), Some("LIKE input"));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_query_rejects_implicit_decimal_float_arithmetic_before_cudf_execution() {
    let dir = unique_temp_dir("cudf-query-engine-native-execution-decimal-float-implicit-reject");
    let path = dir.join("tpch-like.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_tpch_like_types_parquet(&runtime, &path);
    let output_schema = Arc::new(Schema::new(vec![Field::new("bad_amount", DataType::Decimal128(15, 2), true)]));
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["amount".to_string()],
    );

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let projection = builder.add_projection(
        source,
        vec![
            Expr::binary(Expr::column("amount"), BinaryOp::Add, Expr::literal(LiteralValue::Float64(1.0)))
                .alias("bad_amount"),
        ],
        output_schema,
    );
    let plan = builder.finish_with_host_sink(projection);
    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime));

    let error = match execute_query_plan(&context, &plan) {
        Ok(_) => panic!("native plan validation should reject implicit Decimal128/Float64 arithmetic"),
        Err(error) => error,
    };
    assert_eq!(error.code(), QueryErrorCode::UnsupportedType);
    assert_eq!(error.operation(), "cudf_query_engine.plan.validate_expr");
    assert_eq!(error.fact_value("context"), Some("binary arithmetic operands"));
    assert_eq!(error.fact_value("left_type"), Some("Decimal128(15, 2)"));
    assert_eq!(error.fact_value("right_type"), Some("Float64"));
    assert!(error.fact_value("path").is_none());

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_query_rejects_date64_temporal_scalars_before_source_read() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = Arc::new(Schema::new(vec![Field::new("event_date64", DataType::Date64, true)]));

    let cases = [
        (
            "extract_date64",
            vec![Expr::extract(cudf_query_engine::DatePart::Year, Expr::column("event_date64")).alias("event_year")],
            Arc::new(Schema::new(vec![Field::new("event_year", DataType::Int32, true)])),
        ),
        (
            "interval_date64",
            vec![
                Expr::binary(
                    Expr::column("event_date64"),
                    BinaryOp::Add,
                    Expr::literal(LiteralValue::IntervalMonthDayNano { months: 0, days: 1, nanoseconds: 0 }),
                )
                .alias("event_plus_one_day"),
            ],
            Arc::new(Schema::new(vec![Field::new(
                "event_plus_one_day",
                DataType::Timestamp(TimeUnit::Millisecond, None),
                true,
            )])),
        ),
    ];

    for (name, projection, output_schema) in cases {
        let source = LocalParquetSource::new(
            format!("/tmp/does-not-need-to-exist-for-date64-{name}.parquet"),
            Arc::clone(&input_schema),
            vec!["event_date64".to_string()],
        );
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(source));
        let projection = builder.add_projection(source, projection, output_schema);
        let plan = builder.finish_with_host_sink(projection);
        let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime));

        let error = match execute_query_plan(&context, &plan) {
            Ok(_) => panic!("{name} should reject Date64 before source read"),
            Err(error) => error,
        };
        assert_eq!(error.code(), QueryErrorCode::UnsupportedType, "{name}");
        assert!(error.fact_value("path").is_none(), "{name} should fail before source read");
    }
}

#[test]
fn native_query_rejects_out_of_envelope_substring_before_source_read() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = Arc::new(Schema::new(vec![
        Field::new("orderkey", DataType::Int64, false),
        Field::new("status", DataType::Utf8, true),
    ]));

    let cases = [
        (
            "zero_start",
            Expr::substring(Expr::column("status"), 0, Some(1)).alias("bad_substring"),
            QueryErrorCode::UnsupportedExpression,
            None,
        ),
        (
            "negative_length",
            Expr::substring(Expr::column("status"), 1, Some(-1)).alias("bad_substring"),
            QueryErrorCode::UnsupportedExpression,
            None,
        ),
        (
            "stop_overflow",
            Expr::substring(Expr::column("status"), i32::MAX, Some(2)).alias("bad_substring"),
            QueryErrorCode::UnsupportedExpression,
            None,
        ),
        (
            "non_string_source",
            Expr::substring(Expr::column("orderkey"), 1, Some(1)).alias("bad_substring"),
            QueryErrorCode::UnsupportedType,
            Some("SUBSTRING input"),
        ),
    ];

    for (name, expr, expected_code, expected_context) in cases {
        let source = LocalParquetSource::new(
            format!("/tmp/does-not-need-to-exist-for-substring-{name}.parquet"),
            Arc::clone(&input_schema),
            vec!["orderkey".to_string(), "status".to_string()],
        );
        let output_schema = Arc::new(Schema::new(vec![Field::new("bad_substring", DataType::Utf8, true)]));

        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(source));
        let projection = builder.add_projection(source, vec![expr], output_schema);
        let plan = builder.finish_with_host_sink(projection);
        let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime));

        let error = match execute_query_plan(&context, &plan) {
            Ok(_) => panic!("{name} should reject before source read"),
            Err(error) => error,
        };
        assert_eq!(error.code(), expected_code, "{name}");
        assert_eq!(error.operation(), "cudf_query_engine.plan.validate_expr", "{name}");
        assert_eq!(error.fact_value("context"), expected_context, "{name}");
        assert!(error.fact_value("path").is_none(), "{name} should fail before source read");
    }
}

#[test]
fn native_query_rejects_unread_source_column_in_nested_phase2_scalar_before_cudf_execution() {
    let dir = unique_temp_dir("cudf-query-engine-native-execution-phase2-source-column-reject");
    let path = dir.join("tpch-like.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_phase2_scalar_expression_parquet(&runtime, &path);
    let output_schema = Arc::new(Schema::new(vec![Field::new("status_label", DataType::Utf8, true)]));
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["orderkey".to_string()],
    );
    let label = Expr::case(
        vec![cudf_query_engine::CaseWhen::new(
            Expr::like(Expr::column("status"), "O%"),
            Expr::literal(LiteralValue::Utf8("open".to_string())),
        )],
        Expr::literal(LiteralValue::Utf8("other".to_string())),
    )
    .alias("status_label");

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let projection = builder.add_projection(source, vec![label], output_schema);
    let plan = builder.finish_with_host_sink(projection);
    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime));

    let error = match execute_query_plan(&context, &plan) {
        Ok(_) => panic!("native plan validation should reject unread nested scalar columns before cuDF execution"),
        Err(error) => error,
    };
    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.operation(), "cudf_query_engine.plan.validate_expr");
    assert_eq!(error.fact_value("column"), Some("status"));
    assert!(error.fact_value("path").is_none());

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_query_executes_filter_projection_and_group_by_on_partition_columns() {
    let dir = unique_temp_dir("cudf-query-engine-native-execution-partitions");
    let first_path = dir.join("part-001.parquet");
    let second_path = dir.join("part-002.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let file_schema = write_input_parquet(&runtime, &first_path);
    write_input_parquet(&runtime, &second_path);
    let plan = partition_column_query_plan(&first_path, &second_path, file_schema);

    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime));
    let result = execute_query_plan(&context, &plan).unwrap();

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 1);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[2]);
    assert_eq!(int64_array(batch, 1).values().as_ref(), &[1000]);
    assert_eq!(result.metrics().parquet_files, 2);
    assert_eq!(result.metrics().read_columns, 2);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_executes_partition_columns_per_source_chunk() {
    let dir = unique_temp_dir("cudf-query-engine-native-execution-bounded-partitions");
    let first_path = dir.join("part-001.parquet");
    let second_path = dir.join("part-002.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let file_schema = write_input_parquet(&runtime, &first_path);
    write_input_parquet(&runtime, &second_path);
    let plan = partition_column_query_plan(&first_path, &second_path, file_schema);

    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 8 * 1024 * 1024 * 1024,
            max_source_chunk_bytes: 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .build()
        .unwrap();
    let result = execute_query_plan(&context, &plan).unwrap();

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 1);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[2]);
    assert_eq!(int64_array(batch, 1).values().as_ref(), &[1000]);
    assert_eq!(result.metrics().source_chunks_planned, 2);
    assert_eq!(result.metrics().source_chunks_committed, 2);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_executes_chunked_grouped_sum_and_final_merge() {
    let dir = unique_temp_dir("cudf-query-engine-bounded-native-execution");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_row_grouped_input_parquet(&path);

    let projected_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("x00", DataType::Int64, false),
    ]));
    let aggregated_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("sum_x00", DataType::Int64, true),
    ]));

    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["key".to_string(), "m00".to_string(), "m01".to_string(), "payload".to_string()],
    );
    let filter = Expr::binary(Expr::column("payload"), BinaryOp::GtEq, Expr::literal(LiteralValue::Int64(10)));
    let projection =
        vec![Expr::column("key"), Expr::binary(Expr::column("m00"), BinaryOp::Add, Expr::column("m01")).alias("x00")];
    let aggregate = AggregateSpec::new(
        vec![Expr::column("key")],
        vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("x00"), "sum_x00")],
    );
    let sort = SortSpec::new(vec![
        SortExpr::new(Expr::column("sum_x00"), true, false),
        SortExpr::new(Expr::column("key"), false, false),
    ]);

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let filter = builder.add_filter(source, filter, Arc::clone(&input_schema));
    let projection = builder.add_projection(filter, projection, projected_schema);
    let aggregate = builder.add_aggregate(projection, aggregate, Arc::clone(&aggregated_schema));
    let sort = builder.add_sort(aggregate, sort, Arc::clone(&aggregated_schema));
    let limit = builder.add_limit(sort, LimitSpec::fetch(3), aggregated_schema);
    let plan = builder.finish_with_host_sink(limit);

    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 8 * 1024 * 1024 * 1024,
            max_source_chunk_bytes: 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .build()
        .unwrap();
    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.phase_c_memory_control_policy.as_str(), "bounded");
    assert_eq!(metrics.source_chunks_planned, 3);
    assert_eq!(metrics.source_chunks_attempted, 3);
    assert_eq!(metrics.source_chunks_committed, 3);
    assert_eq!(metrics.row_groups_planned, 3);
    assert_eq!(metrics.row_groups_committed, 3);
    assert_eq!(metrics.partial_aggregate_count, 3);
    assert_eq!(metrics.final_merge_attempts, 1);
    assert_eq!(metrics.final_merge_levels, 1);
    assert_eq!(metrics.parquet_files, 1);
    assert_eq!(metrics.row_groups_selected, 3);
    assert_eq!(metrics.read_columns, 4);
    assert_eq!(metrics.input_batches, 3);
    assert_eq!(metrics.input_rows, 6);
    assert_eq!(metrics.boundary.host_to_device_uploads, 0);
    assert_eq!(metrics.boundary.device_to_host_materializations, 1);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 3);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[2, 1, 3]);
    assert_eq!(int64_array(batch, 1).values().as_ref(), &[707, 404, 404]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_executes_min_max_count_and_final_merge() {
    let dir = unique_temp_dir("cudf-query-engine-bounded-native-aggregate-parity");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_row_grouped_input_parquet(&path);

    let aggregated_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("min_m00", DataType::Int64, true),
        Field::new("max_m01", DataType::Int64, true),
        Field::new("count_payload", DataType::Int64, false),
    ]));

    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["key".to_string(), "payload".to_string(), "m00".to_string(), "m01".to_string()],
    );
    let filter = Expr::binary(Expr::column("payload"), BinaryOp::GtEq, Expr::literal(LiteralValue::Int64(10)));
    let aggregate = AggregateSpec::new(
        vec![Expr::column("key")],
        vec![
            AggregateExpr::new(AggregateFunction::Min, Expr::column("m00"), "min_m00"),
            AggregateExpr::new(AggregateFunction::Max, Expr::column("m01"), "max_m01"),
            AggregateExpr::new(AggregateFunction::Count, Expr::column("payload"), "count_payload"),
        ],
    );
    let sort = SortSpec::new(vec![SortExpr::new(Expr::column("key"), false, false)]);

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let filter = builder.add_filter(source, filter, Arc::clone(&input_schema));
    let aggregate = builder.add_aggregate(filter, aggregate, Arc::clone(&aggregated_schema));
    let sort = builder.add_sort(aggregate, sort, Arc::clone(&aggregated_schema));
    let limit = builder.add_limit(sort, LimitSpec::fetch(3), aggregated_schema);
    let plan = builder.finish_with_host_sink(limit);

    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 8 * 1024 * 1024 * 1024,
            max_source_chunk_bytes: 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .build()
        .unwrap();
    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.source_chunks_planned, 3);
    assert_eq!(metrics.source_chunks_committed, 3);
    assert_eq!(metrics.partial_aggregate_count, 3);
    assert_eq!(metrics.final_merge_attempts, 1);
    assert_eq!(metrics.boundary.host_to_device_uploads, 0);
    assert_eq!(metrics.boundary.device_to_host_materializations, 1);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 3);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[1, 2, 3]);
    assert_eq!(int64_array(batch, 1).values().as_ref(), &[100, 200, 400]);
    assert_eq!(int64_array(batch, 2).values().as_ref(), &[3, 5, 4]);
    assert_eq!(int64_array(batch, 3).values().as_ref(), &[2, 2, 1]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_casts_single_partial_grouped_count_to_output_schema() {
    let dir = unique_temp_dir("cudf-query-engine-bounded-native-single-partial-count");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_row_grouped_input_parquet(&path);

    let aggregated_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("count_payload", DataType::Int64, false),
    ]));

    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["key".to_string(), "payload".to_string()],
    );
    let filter = Expr::binary(Expr::column("payload"), BinaryOp::GtEq, Expr::literal(LiteralValue::Int64(10)));
    let aggregate = AggregateSpec::new(
        vec![Expr::column("key")],
        vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("payload"), "count_payload")],
    );
    let sort = SortSpec::new(vec![SortExpr::new(Expr::column("key"), false, false)]);

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let filter = builder.add_filter(source, filter, Arc::clone(&input_schema));
    let aggregate = builder.add_aggregate(filter, aggregate, Arc::clone(&aggregated_schema));
    let sort = builder.add_sort(aggregate, sort, Arc::clone(&aggregated_schema));
    let plan = builder.finish_with_host_sink(sort);

    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 8 * 1024 * 1024 * 1024,
            max_source_chunk_bytes: 8 * 1024 * 1024 * 1024,
            max_row_groups_per_chunk: 16,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .build()
        .unwrap();
    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.source_chunks_planned, 1);
    assert_eq!(metrics.partial_aggregate_count, 1);
    assert_eq!(metrics.final_merge_attempts, 0);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.column(1).data_type(), &DataType::Int64);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[1, 2, 3]);
    assert_eq!(int64_array(batch, 1).values().as_ref(), &[2, 2, 1]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_executes_global_sum_min_max_count_and_final_merge() {
    let dir = unique_temp_dir("cudf-query-engine-bounded-native-global-aggregate-parity");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_row_grouped_input_parquet(&path);

    let aggregated_schema = Arc::new(Schema::new(vec![
        Field::new("sum_m00", DataType::Int64, true),
        Field::new("min_m00", DataType::Int64, true),
        Field::new("max_m01", DataType::Int64, true),
        Field::new("count_payload", DataType::Int64, false),
    ]));

    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["payload".to_string(), "m00".to_string(), "m01".to_string()],
    );
    let filter = Expr::binary(Expr::column("payload"), BinaryOp::GtEq, Expr::literal(LiteralValue::Int64(10)));
    let aggregate = AggregateSpec::new(
        Vec::new(),
        vec![
            AggregateExpr::new(AggregateFunction::Sum, Expr::column("m00"), "sum_m00"),
            AggregateExpr::new(AggregateFunction::Min, Expr::column("m00"), "min_m00"),
            AggregateExpr::new(AggregateFunction::Max, Expr::column("m01"), "max_m01"),
            AggregateExpr::new(AggregateFunction::Count, Expr::column("payload"), "count_payload"),
        ],
    );

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let filter = builder.add_filter(source, filter, Arc::clone(&input_schema));
    let aggregate = builder.add_aggregate(filter, aggregate, aggregated_schema);
    let plan = builder.finish_with_host_sink(aggregate);

    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 8 * 1024 * 1024 * 1024,
            max_source_chunk_bytes: 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .build()
        .unwrap();
    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.source_chunks_planned, 3);
    assert_eq!(metrics.source_chunks_committed, 3);
    assert_eq!(metrics.partial_aggregate_count, 3);
    assert_eq!(metrics.final_merge_attempts, 1);
    assert_eq!(metrics.boundary.host_to_device_uploads, 0);
    assert_eq!(metrics.boundary.device_to_host_materializations, 1);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 1);
    assert_eq!(int64_array(batch, 0).values().as_ref(), &[1500]);
    assert_eq!(int64_array(batch, 1).values().as_ref(), &[100]);
    assert_eq!(int64_array(batch, 2).values().as_ref(), &[5]);
    assert_eq!(int64_array(batch, 3).values().as_ref(), &[5]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_finalizes_single_chunk_global_sum_min_max_hidden_state() {
    let dir = unique_temp_dir("cudf-query-engine-bounded-native-single-chunk-global-state");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_row_grouped_input_parquet(&path);
    let expected_names = ["sum_m00", "min_m00", "max_m01", "count_payload"];

    let aggregated_schema = Arc::new(Schema::new(vec![
        Field::new("sum_m00", DataType::Int64, true),
        Field::new("min_m00", DataType::Int64, true),
        Field::new("max_m01", DataType::Int64, true),
        Field::new("count_payload", DataType::Int64, false),
    ]));
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["payload".to_string(), "m00".to_string(), "m01".to_string()],
    );
    let aggregate = AggregateSpec::new(
        Vec::new(),
        vec![
            AggregateExpr::new(AggregateFunction::Sum, Expr::column("m00"), "sum_m00"),
            AggregateExpr::new(AggregateFunction::Min, Expr::column("m00"), "min_m00"),
            AggregateExpr::new(AggregateFunction::Max, Expr::column("m01"), "max_m01"),
            AggregateExpr::new(AggregateFunction::Count, Expr::column("payload"), "count_payload"),
        ],
    );

    let build_plan = |host_sink: bool| {
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(source.clone()));
        let filter = Expr::binary(Expr::column("payload"), BinaryOp::GtEq, Expr::literal(LiteralValue::Int64(10)));
        let filter = builder.add_filter(source, filter, Arc::clone(&input_schema));
        let aggregate = builder.add_aggregate(filter, aggregate.clone(), Arc::clone(&aggregated_schema));
        if host_sink { builder.finish_with_host_sink(aggregate) } else { builder.finish_gpu_resident(aggregate) }
    };
    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 8 * 1024 * 1024 * 1024,
            max_source_chunk_bytes: 1024 * 1024,
            max_row_groups_per_chunk: 8,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .build()
        .unwrap();

    let host_result = execute_query_plan(&context, &build_plan(true)).unwrap();
    assert_eq!(host_result.metrics().source_chunks_planned, 1);
    assert_eq!(host_result.metrics().partial_aggregate_count, 1);
    assert_eq!(host_result.metrics().final_merge_attempts, 0);
    let QueryOutput::HostArrow(batch) = host_result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_columns(), expected_names.len());
    assert_eq!(batch.schema().fields().iter().map(|field| field.name().as_str()).collect::<Vec<_>>(), expected_names);
    assert!(batch.schema().fields().iter().all(|field| !field.name().starts_with("__cudf_global_valid_count_")));
    assert_eq!(batch.num_rows(), 1);
    assert_eq!(int64_array(batch, 0).values().as_ref(), &[1500]);
    assert_eq!(int64_array(batch, 1).values().as_ref(), &[100]);
    assert_eq!(int64_array(batch, 2).values().as_ref(), &[5]);
    assert_eq!(int64_array(batch, 3).values().as_ref(), &[5]);

    let gpu_result = execute_query_plan(&context, &build_plan(false)).unwrap();
    let QueryOutput::GpuResident(frame) = gpu_result.output() else {
        panic!("gpu sink should return a GPU-resident frame");
    };
    assert_eq!(frame.num_columns(), expected_names.len());
    assert_eq!(frame.frame().column_names(), expected_names);
    assert_eq!(frame.schema().fields().iter().map(|field| field.name().as_str()).collect::<Vec<_>>(), expected_names);
    assert_eq!(frame.device_placement(), context.device_placement());
    assert_eq!(frame.runtime().device().ordinal(), runtime.device().ordinal());
    assert_eq!(gpu_result.metrics().boundary.device_to_host_materializations, 0);

    let mut materialization_metrics = QueryMetrics::default();
    let materialized = materialize_gpu_result_to_host(frame.clone(), &mut materialization_metrics).unwrap();
    assert_eq!(materialization_metrics.boundary.device_to_host_materializations, 1);
    assert_eq!(materialization_metrics.boundary.device_to_host_materialized_rows, 1);
    assert_eq!(
        materialized.schema().fields().iter().map(|field| field.name().as_str()).collect::<Vec<_>>(),
        expected_names
    );
    assert_eq!(int64_array(&materialized, 0).values().as_ref(), &[1500]);

    let output_schema = Arc::new(Schema::new(vec![Field::new("sum_m00", DataType::Int64, true)]));
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new(
        "q_global_partial",
        Arc::clone(&aggregated_schema),
    )));
    let projection = builder.add_projection(source, vec![Expr::column("sum_m00")], Arc::clone(&output_schema));
    let consumer_plan = builder.finish_with_host_sink(projection);
    let mut gpu_inputs = GpuResidentInputMap::new();
    gpu_inputs.insert("q_global_partial".to_string(), frame.clone());

    let consumed = execute_query_plan_with_gpu_inputs(&context, &consumer_plan, &gpu_inputs).unwrap();
    assert_eq!(consumed.metrics().boundary.gpu_resident_input_handoffs, 1);
    assert_eq!(consumed.metrics().boundary.host_to_device_uploads, 0);
    assert_eq!(consumed.metrics().boundary.device_to_host_materializations, 1);
    assert_eq!(consumed.metrics().parquet_files, 0);
    let QueryOutput::HostArrow(batch) = consumed.output() else {
        panic!("GPU-resident source consumer should materialize only at the final host sink");
    };
    assert_eq!(batch.schema(), output_schema);
    assert_eq!(int64_array(batch, 0).values().as_ref(), &[1500]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_executes_global_avg_from_sum_count_state() {
    let dir = unique_temp_dir("cudf-query-engine-bounded-native-global-avg");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_uneven_global_avg_input_parquet(&path);

    let aggregated_schema = Arc::new(Schema::new(vec![Field::new("avg_m00", DataType::Float64, true)]));
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["m00".to_string()],
    );
    let aggregate = AggregateSpec::new(
        Vec::new(),
        vec![AggregateExpr::new(AggregateFunction::Avg, Expr::column("m00"), "avg_m00")],
    );

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let aggregate = builder.add_aggregate(source, aggregate, aggregated_schema);
    let plan = builder.finish_with_host_sink(aggregate);

    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 8 * 1024 * 1024 * 1024,
            max_source_chunk_bytes: 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .build()
        .unwrap();
    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.source_chunks_planned, 2);
    assert_eq!(metrics.source_chunks_committed, 2);
    assert_eq!(metrics.partial_aggregate_count, 2);
    assert_eq!(metrics.final_merge_attempts, 1);
    assert_eq!(metrics.boundary.host_to_device_uploads, 0);
    assert_eq!(metrics.boundary.device_to_host_materializations, 1);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 1);
    assert_eq!(float64_array(batch, 0).value(0), 72.0);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_executes_global_null_on_empty_sum_avg_for_all_null_input() {
    let dir = unique_temp_dir("cudf-query-engine-bounded-native-global-all-null");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_nullable_global_aggregate_input_parquet(&path);

    let plan = nullable_global_aggregate_plan(&path, Arc::clone(&input_schema), None);
    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 8 * 1024 * 1024 * 1024,
            max_source_chunk_bytes: 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .build()
        .unwrap();
    let result = execute_query_plan(&context, &plan).unwrap();
    assert_eq!(result.metrics().source_chunks_planned, 2);
    assert_eq!(result.metrics().source_chunks_committed, 2);
    assert_eq!(result.metrics().partial_aggregate_count, 2);
    assert_eq!(result.metrics().final_merge_attempts, 1);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 1);
    assert!(int64_array(batch, 0).is_null(0));
    assert!(float64_array(batch, 1).is_null(0));
    assert!(int64_array(batch, 2).is_null(0));
    assert!(int64_array(batch, 3).is_null(0));
    assert_eq!(int64_array(batch, 4).value(0), 0);
    assert_eq!(int64_array(batch, 5).value(0), 4);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_executes_global_null_on_empty_sum_avg_for_filtered_empty_input() {
    let dir = unique_temp_dir("cudf-query-engine-bounded-native-global-filtered-empty");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_nullable_global_aggregate_input_parquet(&path);
    let filter = Expr::binary(Expr::column("payload"), BinaryOp::Gt, Expr::literal(LiteralValue::Int64(100)));

    let plan = nullable_global_aggregate_plan(&path, Arc::clone(&input_schema), Some(filter));
    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 8 * 1024 * 1024 * 1024,
            max_source_chunk_bytes: 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .build()
        .unwrap();
    let result = execute_query_plan(&context, &plan).unwrap();
    assert_eq!(result.metrics().source_chunks_planned, 2);
    assert_eq!(result.metrics().source_chunks_committed, 2);
    assert_eq!(result.metrics().partial_aggregate_count, 2);
    assert_eq!(result.metrics().final_merge_attempts, 1);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 1);
    assert!(int64_array(batch, 0).is_null(0));
    assert!(float64_array(batch, 1).is_null(0));
    assert!(int64_array(batch, 2).is_null(0));
    assert!(int64_array(batch, 3).is_null(0));
    assert_eq!(int64_array(batch, 4).value(0), 0);
    assert_eq!(int64_array(batch, 5).value(0), 0);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_executes_decimal_global_avg_from_sum_count_state() {
    let dir = unique_temp_dir("cudf-query-engine-bounded-native-decimal-global-avg");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_decimal_avg_input_parquet(&path);

    let aggregated_schema = Arc::new(Schema::new(vec![Field::new("avg_amount", DataType::Decimal128(19, 6), true)]));
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["amount".to_string()],
    );
    let aggregate = AggregateSpec::new(
        Vec::new(),
        vec![AggregateExpr::new(AggregateFunction::Avg, Expr::column("amount"), "avg_amount")],
    );

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let aggregate = builder.add_aggregate(source, aggregate, aggregated_schema);
    let plan = builder.finish_with_host_sink(aggregate);

    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 8 * 1024 * 1024 * 1024,
            max_source_chunk_bytes: 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .build()
        .unwrap();
    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.source_chunks_planned, 3);
    assert_eq!(metrics.source_chunks_committed, 3);
    assert_eq!(metrics.partial_aggregate_count, 3);
    assert_eq!(metrics.final_merge_attempts, 1);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 1);
    let avg_amount = decimal128_array(batch, 0);
    assert_eq!(avg_amount.value(0), 2_000_000);
    assert_eq!(avg_amount.precision(), 19);
    assert_eq!(avg_amount.scale(), 6);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_executes_q6_shaped_decimal_global_sum() {
    let dir = unique_temp_dir("cudf-query-engine-bounded-native-q6-shaped");
    let path = dir.join("lineitem.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_q6_lineitem_parquet(&path);
    let projected_schema =
        Arc::new(Schema::new(vec![Field::new("__cudf_aggregate_arg_0", DataType::Decimal128(38, 4), true)]));
    let aggregated_schema = Arc::new(Schema::new(vec![Field::new("revenue", DataType::Decimal128(38, 4), true)]));

    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec![
            "l_discount".to_string(),
            "l_extendedprice".to_string(),
            "l_quantity".to_string(),
            "l_shipdate".to_string(),
        ],
    );
    let filter = Expr::binary(
        Expr::binary(
            Expr::binary(
                Expr::binary(
                    Expr::column("l_shipdate"),
                    BinaryOp::GtEq,
                    Expr::literal(LiteralValue::Date32(date32(1994, 1, 1))),
                ),
                BinaryOp::And,
                Expr::binary(
                    Expr::column("l_shipdate"),
                    BinaryOp::Lt,
                    Expr::literal(LiteralValue::Date32(date32(1995, 1, 1))),
                ),
            ),
            BinaryOp::And,
            Expr::binary(
                Expr::binary(
                    Expr::column("l_discount"),
                    BinaryOp::GtEq,
                    Expr::literal(LiteralValue::Decimal128 { raw: 5, precision: 15, scale: 2 }),
                ),
                BinaryOp::And,
                Expr::binary(
                    Expr::column("l_discount"),
                    BinaryOp::LtEq,
                    Expr::literal(LiteralValue::Decimal128 { raw: 7, precision: 15, scale: 2 }),
                ),
            ),
        ),
        BinaryOp::And,
        Expr::binary(
            Expr::column("l_quantity"),
            BinaryOp::Lt,
            Expr::literal(LiteralValue::Decimal128 { raw: 2_400, precision: 15, scale: 2 }),
        ),
    );
    let projection = vec![
        Expr::binary(Expr::column("l_extendedprice"), BinaryOp::Multiply, Expr::column("l_discount"))
            .alias("__cudf_aggregate_arg_0"),
    ];
    let aggregate = AggregateSpec::new(
        Vec::new(),
        vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("__cudf_aggregate_arg_0"), "revenue")],
    );

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let filter = builder.add_filter(source, filter, Arc::clone(&input_schema));
    let projection = builder.add_projection(filter, projection, projected_schema);
    let aggregate = builder.add_aggregate(projection, aggregate, aggregated_schema);
    let plan = builder.finish_with_host_sink(aggregate);

    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 8 * 1024 * 1024 * 1024,
            max_source_chunk_bytes: 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .build()
        .unwrap();
    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.source_chunks_planned, 4);
    assert_eq!(metrics.source_chunks_committed, 4);
    assert_eq!(metrics.partial_aggregate_count, 4);
    assert_eq!(metrics.final_merge_attempts, 1);
    assert_eq!(metrics.boundary.host_to_device_uploads, 0);
    assert_eq!(metrics.boundary.device_to_host_materializations, 1);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 1);
    let revenue = decimal128_array(batch, 0);
    assert_eq!(revenue.value(0), 260_000);
    assert_eq!(revenue.scale(), 4);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_zero_column_count_uses_row_group_metadata_without_read_columns() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let aggregated_schema = Arc::new(Schema::new(vec![Field::new("count_rows", DataType::Int64, false)]));
    let source = LocalParquetSource::zero_column_count(vec![
        LocalParquetFile::new("/tmp/count-only.parquet")
            .with_stable_id("count-only")
            .with_file_size_bytes(600)
            .with_row_groups(vec![0, 2])
            .with_row_group_bytes(vec![100, 200, 300])
            .with_row_group_rows(vec![5, 7, 11]),
    ]);
    let aggregate = AggregateSpec::new(
        Vec::new(),
        vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column(ZERO_COLUMN_COUNT_COLUMN), "count_rows")],
    );
    let sort = SortSpec::new(vec![SortExpr::new(Expr::column("count_rows"), true, false)]);

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let aggregate = builder.add_aggregate(source, aggregate, Arc::clone(&aggregated_schema));
    let sort = builder.add_sort(aggregate, sort, Arc::clone(&aggregated_schema));
    let limit = builder.add_limit(sort, LimitSpec::fetch(1), aggregated_schema);
    let plan = builder.finish_with_host_sink(limit);

    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 8 * 1024 * 1024 * 1024,
            max_source_chunk_bytes: 250,
            max_row_groups_per_chunk: 8,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .build()
        .unwrap();
    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.source_chunks_planned, 2);
    assert_eq!(metrics.source_chunks_committed, 2);
    assert_eq!(metrics.row_groups_selected, 2);
    assert_eq!(metrics.read_columns, 0);
    assert_eq!(metrics.input_rows, 16);
    assert_eq!(metrics.boundary.host_to_device_uploads, 0);
    assert_eq!(metrics.boundary.device_to_host_materializations, 1);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 1);
    assert_eq!(int64_array(batch, 0).values().as_ref(), &[16]);
}

#[test]
fn bounded_native_query_executes_avg_from_sum_count_state() {
    let dir = unique_temp_dir("cudf-query-engine-bounded-native-avg");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_uneven_avg_input_parquet(&path);

    let aggregated_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("avg_m00", DataType::Float64, true),
    ]));

    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["key".to_string(), "payload".to_string(), "m00".to_string()],
    );
    let filter = Expr::binary(Expr::column("payload"), BinaryOp::GtEq, Expr::literal(LiteralValue::Int64(10)));
    let aggregate = AggregateSpec::new(
        vec![Expr::column("key")],
        vec![AggregateExpr::new(AggregateFunction::Avg, Expr::column("m00"), "avg_m00")],
    );
    let sort = SortSpec::new(vec![
        SortExpr::new(Expr::column("avg_m00"), true, false),
        SortExpr::new(Expr::column("key"), false, false),
    ]);

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let filter = builder.add_filter(source, filter, Arc::clone(&input_schema));
    let aggregate = builder.add_aggregate(filter, aggregate, Arc::clone(&aggregated_schema));
    let sort = builder.add_sort(aggregate, sort, Arc::clone(&aggregated_schema));
    let limit = builder.add_limit(sort, LimitSpec::fetch(3), aggregated_schema);
    let plan = builder.finish_with_host_sink(limit);

    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 8 * 1024 * 1024 * 1024,
            max_source_chunk_bytes: 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .build()
        .unwrap();
    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.source_chunks_planned, 2);
    assert_eq!(metrics.source_chunks_committed, 2);
    assert_eq!(metrics.partial_aggregate_count, 2);
    assert_eq!(metrics.final_merge_attempts, 1);
    assert_eq!(metrics.boundary.host_to_device_uploads, 0);
    assert_eq!(metrics.boundary.device_to_host_materializations, 1);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 3);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[3, 2, 1]);
    assert_eq!(float64_array(batch, 1).values().as_ref(), &[300.0, 150.0, 20.0]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_executes_single_chunk_avg_finalization() {
    let dir = unique_temp_dir("cudf-query-engine-bounded-native-single-chunk-avg");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_uneven_avg_input_parquet(&path);

    let aggregated_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("avg_m00", DataType::Float64, true),
    ]));

    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["key".to_string(), "payload".to_string(), "m00".to_string()],
    );
    let aggregate = AggregateSpec::new(
        vec![Expr::column("key")],
        vec![AggregateExpr::new(AggregateFunction::Avg, Expr::column("m00"), "avg_m00")],
    );
    let sort = SortSpec::new(vec![SortExpr::new(Expr::column("key"), false, false)]);

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let aggregate = builder.add_aggregate(source, aggregate, Arc::clone(&aggregated_schema));
    let sort = builder.add_sort(aggregate, sort, Arc::clone(&aggregated_schema));
    let limit = builder.add_limit(sort, LimitSpec::fetch(3), aggregated_schema);
    let plan = builder.finish_with_host_sink(limit);

    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 8 * 1024 * 1024 * 1024,
            max_source_chunk_bytes: 1024 * 1024,
            max_row_groups_per_chunk: 8,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .build()
        .unwrap();
    let result = execute_query_plan(&context, &plan).unwrap();
    assert_eq!(result.metrics().source_chunks_planned, 1);
    assert_eq!(result.metrics().final_merge_attempts, 0);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[1, 2, 3]);
    assert_eq!(float64_array(batch, 1).values().as_ref(), &[20.0, 150.0, 300.0]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_executes_nullable_avg_with_null_all_group() {
    let dir = unique_temp_dir("cudf-query-engine-bounded-native-nullable-avg");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_nullable_avg_input_parquet(&path);

    let aggregated_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("avg_m00", DataType::Float64, true),
    ]));

    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["key".to_string(), "m00".to_string()],
    );
    let aggregate = AggregateSpec::new(
        vec![Expr::column("key")],
        vec![AggregateExpr::new(AggregateFunction::Avg, Expr::column("m00"), "avg_m00")],
    );
    let sort = SortSpec::new(vec![SortExpr::new(Expr::column("key"), false, false)]);

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let aggregate = builder.add_aggregate(source, aggregate, Arc::clone(&aggregated_schema));
    let sort = builder.add_sort(aggregate, sort, Arc::clone(&aggregated_schema));
    let limit = builder.add_limit(sort, LimitSpec::fetch(3), aggregated_schema);
    let plan = builder.finish_with_host_sink(limit);

    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 8 * 1024 * 1024 * 1024,
            max_source_chunk_bytes: 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .build()
        .unwrap();
    let result = execute_query_plan(&context, &plan).unwrap();

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    let avg = float64_array(batch, 1);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[1, 2, 3]);
    assert!(avg.is_null(0));
    assert_eq!(avg.value(1), 10.0);
    assert_eq!(avg.value(2), 30.0);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_rejects_avg_with_unsupported_output_schema() {
    let dir = unique_temp_dir("cudf-query-engine-bounded-native-avg-output-type");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_uneven_avg_input_parquet(&path);

    let aggregated_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("avg_m00", DataType::Int64, true),
    ]));

    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["key".to_string(), "m00".to_string()],
    );
    let aggregate = AggregateSpec::new(
        vec![Expr::column("key")],
        vec![AggregateExpr::new(AggregateFunction::Avg, Expr::column("m00"), "avg_m00")],
    );
    let sort = SortSpec::new(vec![SortExpr::new(Expr::column("key"), false, false)]);

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let aggregate = builder.add_aggregate(source, aggregate, Arc::clone(&aggregated_schema));
    let sort = builder.add_sort(aggregate, sort, Arc::clone(&aggregated_schema));
    let limit = builder.add_limit(sort, LimitSpec::fetch(3), aggregated_schema);
    let plan = builder.finish_with_host_sink(limit);

    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 8 * 1024 * 1024 * 1024,
            max_source_chunk_bytes: 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .build()
        .unwrap();

    match execute_query_plan(&context, &plan) {
        Ok(_) => panic!("avg with unsupported output schema should be rejected"),
        Err(error) => {
            assert_eq!(error.code(), QueryErrorCode::UnsupportedType);
            assert_eq!(error.operation(), "cudf_query_engine.aggregate.validate_avg");
            assert_eq!(error.fact_value("alias"), Some("avg_m00"));
        },
    }

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_executes_post_aggregate_projection_before_final_sort() {
    let dir = unique_temp_dir("cudf-query-engine-bounded-post-aggregate-projection");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_row_grouped_input_parquet(&path);
    let aggregated_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("sum_m00", DataType::Int64, true),
    ]));
    let post_projection_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("total", DataType::Int64, true),
    ]));

    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["key".to_string(), "payload".to_string(), "m00".to_string()],
    );
    let filter = Expr::binary(Expr::column("payload"), BinaryOp::GtEq, Expr::literal(LiteralValue::Int64(10)));
    let aggregate = AggregateSpec::new(
        vec![Expr::column("key")],
        vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("m00"), "sum_m00")],
    );
    let projection = vec![Expr::column("key"), Expr::column("sum_m00").alias("total")];
    let sort = SortSpec::new(vec![
        SortExpr::new(Expr::column("total"), true, false),
        SortExpr::new(Expr::column("key"), false, false),
    ]);

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let filter = builder.add_filter(source, filter, Arc::clone(&input_schema));
    let aggregate = builder.add_aggregate(filter, aggregate, aggregated_schema);
    let projection = builder.add_projection(aggregate, projection, Arc::clone(&post_projection_schema));
    let sort = builder.add_sort(projection, sort, Arc::clone(&post_projection_schema));
    let limit = builder.add_limit(sort, LimitSpec::fetch(3), post_projection_schema);
    let plan = builder.finish_with_host_sink(limit);

    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 8 * 1024 * 1024 * 1024,
            max_source_chunk_bytes: 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .build()
        .unwrap();
    let result = execute_query_plan(&context, &plan).unwrap();

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 3);
    assert_eq!(batch.schema().field(1).name(), "total");
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[2, 1, 3]);
    assert_eq!(int64_array(batch, 1).values().as_ref(), &[700, 400, 400]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_reports_reservation_failure_before_chunk_read() {
    let dir = unique_temp_dir("cudf-query-engine-bounded-reservation-failure");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_row_grouped_input_parquet(&path);
    let aggregated_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("sum_m00", DataType::Int64, true),
    ]));

    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["key".to_string(), "m00".to_string()],
    );
    let aggregate = AggregateSpec::new(
        vec![Expr::column("key")],
        vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("m00"), "sum_m00")],
    );

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let aggregate = builder.add_aggregate(source, aggregate, aggregated_schema);
    let plan = builder.finish_with_host_sink(aggregate);
    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 1,
            max_source_chunk_bytes: 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .build()
        .unwrap();

    let error = match execute_query_plan(&context, &plan) {
        Ok(_) => panic!("bounded execution should fail the source reservation"),
        Err(error) => error,
    };
    assert_eq!(error.code(), QueryErrorCode::MemoryReservationFailed);
    assert_eq!(error.fact_value("memory_boundary"), Some("source_chunk_read"));
    assert_eq!(error.fact_value("reservation_kind"), Some("source_read"));
    assert_eq!(error.fact_value("budget_bytes"), Some("1"));
    assert_eq!(error.fact_value("chunk_index"), Some("0"));
    assert_eq!(error.fact_value("file_index"), Some("0"));
    assert_eq!(error.fact_value("row_groups"), Some("0"));
    assert_eq!(error.fact_value("stage"), Some("source_chunk_read"));
    assert_eq!(error.fact_value("attempt"), Some("1"));
    assert_eq!(error.fact_value("max_attempts"), Some("1"));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_reports_retry_facts_when_chunk_retry_exhausts() {
    let dir = unique_temp_dir("cudf-query-engine-bounded-chunk-retry-exhausted");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_row_grouped_input_parquet(&path);
    let aggregated_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("sum_m00", DataType::Int64, true),
    ]));

    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["key".to_string(), "m00".to_string()],
    );
    let aggregate = AggregateSpec::new(
        vec![Expr::column("key")],
        vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("m00"), "sum_m00")],
    );

    let mut builder = QueryPlanBuilder::new();
    let source_id = builder.add_source(SourceNode::LocalParquet(source));
    let aggregate = builder.add_aggregate(source_id, aggregate, aggregated_schema);
    let plan = builder.finish_with_host_sink(aggregate);
    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 1,
            max_source_chunk_bytes: 1024 * 1024,
            max_row_groups_per_chunk: 3,
            max_retry_attempts: 2,
            min_retry_chunk_row_groups: 1,
        })
        .build()
        .unwrap();

    let error = match execute_query_plan(&context, &plan) {
        Ok(_) => panic!("bounded execution should fail after exhausting source chunk retries"),
        Err(error) => error,
    };

    assert_eq!(error.code(), QueryErrorCode::MemoryReservationFailed);
    assert_eq!(error.fact_value("memory_boundary"), Some("source_chunk_read"));
    assert_eq!(error.fact_value("reservation_kind"), Some("source_read"));
    assert_eq!(error.fact_value("stage"), Some("source_chunk_read"));
    assert_eq!(error.fact_value("chunk_index"), Some("0"));
    assert_eq!(error.fact_value("file_index"), Some("0"));
    assert_eq!(error.fact_value("row_groups"), Some("0"));
    assert_eq!(error.fact_value("attempt"), Some("2"));
    assert_eq!(error.fact_value("max_attempts"), Some("2"));
    assert_eq!(error.fact_value("retry_action"), Some("fail_min_chunk"));
    assert_eq!(error.fact_value("resume_boundary"), Some("source_chunk"));
    assert_eq!(error.fact_value("committed_chunks"), Some("0"));
    assert_eq!(error.fact_value("committed_partials"), Some("0"));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_reports_reservation_failure_before_host_materialization() {
    let dir = unique_temp_dir("cudf-query-engine-bounded-materialization-reservation-failure");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_single_row_group_final_sort_input_parquet(&path, 4096);
    let aggregated_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("sum_m00", DataType::Int64, true),
    ]));

    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["key".to_string(), "m00".to_string()],
    );
    let aggregate = AggregateSpec::new(
        vec![Expr::column("key")],
        vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("m00"), "sum_m00")],
    );

    let mut builder = QueryPlanBuilder::new();
    let source_id = builder.add_source(SourceNode::LocalParquet(source.clone()));
    let one_chunk_plan =
        plan_local_parquet_source_chunks(source_id, &source, SourceChunkPlanningPolicy::new(None, 1)).unwrap();
    assert_eq!(one_chunk_plan.chunk_count(), 1);
    let source_chunk_bytes = one_chunk_plan.chunks[0].known_row_group_bytes.unwrap_or(0);
    let aggregate = builder.add_aggregate(source_id, aggregate, Arc::clone(&aggregated_schema));
    let plan = builder.finish_with_host_sink(aggregate);

    let baseline_context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 8 * 1024 * 1024 * 1024,
            max_source_chunk_bytes: 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .build()
        .unwrap();
    let baseline = execute_query_plan(&baseline_context, &plan).unwrap();
    let resident_peak = baseline.metrics().resident_dataframe_peak_bytes.unwrap();
    let constrained_budget = source_chunk_bytes.saturating_add(resident_peak).saturating_add(4096);
    assert!(
        constrained_budget < resident_peak.saturating_mul(2),
        "test data must leave a budget window where execution fits but materialization reservation fails"
    );

    let constrained_context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: constrained_budget,
            max_source_chunk_bytes: 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .build()
        .unwrap();

    let error = match execute_query_plan(&constrained_context, &plan) {
        Ok(_) => panic!("bounded execution should fail the host materialization reservation"),
        Err(error) => error,
    };
    assert_eq!(error.code(), QueryErrorCode::MemoryReservationFailed);
    assert_eq!(error.fact_value("memory_boundary"), Some("host_materialization"));
    assert_eq!(error.fact_value("reservation_kind"), Some("materialization"));
    assert_eq!(error.fact_value("node_id"), Some("n1"));
    assert_eq!(error.fact_value("stage"), Some("materialization"));
    assert_eq!(error.fact_value("attempt"), Some("1"));
    assert_eq!(error.fact_value("max_attempts"), Some("1"));
    assert_eq!(error.fact_value("resume_boundary"), Some("host_materialization"));
    assert_eq!(error.fact_value("committed_chunks"), Some("1"));
    assert_eq!(error.fact_value("committed_partials"), Some("1"));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_retries_chunk_after_source_reservation_failure() {
    let dir = unique_temp_dir("cudf-query-engine-bounded-chunk-retry");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_row_grouped_input_parquet(&path);

    let projected_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("x00", DataType::Int64, false),
    ]));
    let aggregated_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("sum_x00", DataType::Int64, true),
    ]));

    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["key".to_string(), "m00".to_string(), "m01".to_string(), "payload".to_string()],
    );
    let filter = Expr::binary(Expr::column("payload"), BinaryOp::GtEq, Expr::literal(LiteralValue::Int64(10)));
    let projection =
        vec![Expr::column("key"), Expr::binary(Expr::column("m00"), BinaryOp::Add, Expr::column("m01")).alias("x00")];
    let aggregate = AggregateSpec::new(
        vec![Expr::column("key")],
        vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("x00"), "sum_x00")],
    );
    let sort = SortSpec::new(vec![
        SortExpr::new(Expr::column("sum_x00"), true, false),
        SortExpr::new(Expr::column("key"), false, false),
    ]);

    let mut builder = QueryPlanBuilder::new();
    let source_id = builder.add_source(SourceNode::LocalParquet(source.clone()));
    let one_chunk_plan =
        plan_local_parquet_source_chunks(source_id, &source, SourceChunkPlanningPolicy::new(None, 3)).unwrap();
    assert_eq!(one_chunk_plan.chunk_count(), 1);
    let initial_chunk_bytes = one_chunk_plan.chunks[0].known_row_group_bytes.unwrap();
    let filter = builder.add_filter(source_id, filter, Arc::clone(&input_schema));
    let projection = builder.add_projection(filter, projection, projected_schema);
    let aggregate = builder.add_aggregate(projection, aggregate, Arc::clone(&aggregated_schema));
    let sort = builder.add_sort(aggregate, sort, Arc::clone(&aggregated_schema));
    let limit = builder.add_limit(sort, LimitSpec::fetch(3), aggregated_schema);
    let plan = builder.finish_with_host_sink(limit);

    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: initial_chunk_bytes.saturating_sub(1),
            max_source_chunk_bytes: initial_chunk_bytes,
            max_row_groups_per_chunk: 3,
            max_retry_attempts: 3,
            min_retry_chunk_row_groups: 1,
        })
        .build()
        .unwrap();

    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.source_chunks_planned, 1);
    assert_eq!(metrics.source_chunks_retried, 1);
    assert_eq!(metrics.retry_count, 1);
    assert_eq!(metrics.retry_reservation_failure_count, 1);
    assert_eq!(metrics.reservation_denied_count, 1);
    assert_eq!(metrics.source_chunks_attempted, 3);
    assert_eq!(metrics.source_chunks_committed, 2);
    assert_eq!(metrics.row_groups_committed, 3);
    assert_eq!(metrics.partial_aggregate_count, 2);
    assert_eq!(metrics.final_merge_attempts, 1);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 3);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[2, 1, 3]);
    assert_eq!(int64_array(batch, 1).values().as_ref(), &[707, 404, 404]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_retries_final_merge_with_reduced_fan_in() {
    let dir = unique_temp_dir("cudf-query-engine-bounded-final-merge-retry");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_final_merge_retry_input_parquet(&path, 1024);
    let aggregated_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("sum_m00", DataType::Int64, true),
    ]));

    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["key".to_string(), "m00".to_string()],
    );
    let aggregate = AggregateSpec::new(
        vec![Expr::column("key")],
        vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("m00"), "sum_m00")],
    );
    let sort = SortSpec::new(vec![SortExpr::new(Expr::column("key"), false, false)]);

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let aggregate = builder.add_aggregate(source, aggregate, Arc::clone(&aggregated_schema));
    let sort = builder.add_sort(aggregate, sort, Arc::clone(&aggregated_schema));
    let limit = builder.add_limit(sort, LimitSpec::fetch(5), aggregated_schema);
    let plan = builder.finish_with_host_sink(limit);

    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 65_000,
            max_source_chunk_bytes: 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 3,
            min_retry_chunk_row_groups: 1,
        })
        .build()
        .unwrap();

    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.source_chunks_planned, 3);
    assert_eq!(metrics.source_chunks_committed, 3);
    assert_eq!(metrics.final_merge_attempts, 3);
    assert_eq!(metrics.final_merge_levels, 2);
    assert_eq!(metrics.retry_count, 1);
    assert_eq!(metrics.retry_reservation_failure_count, 1);
    assert_eq!(metrics.reservation_denied_count, 1);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 5);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[0, 1, 2, 3, 4]);
    assert_eq!(int64_array(batch, 1).values().as_ref(), &[3, 3, 3, 3, 3]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_retries_global_final_merge_with_consistent_state_schema() {
    let dir = unique_temp_dir("cudf-query-engine-bounded-global-final-merge-retry");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_final_merge_retry_input_parquet(&path, 1024);
    let repeated_sums = 1024;
    let mut fields = (0..repeated_sums)
        .map(|index| Field::new(format!("sum_m00_{index}"), DataType::Int64, true))
        .collect::<Vec<_>>();
    fields.extend([
        Field::new("min_m00", DataType::Int64, true),
        Field::new("max_m00", DataType::Int64, true),
        Field::new("count_m00", DataType::Int64, false),
    ]);
    let aggregated_schema = Arc::new(Schema::new(fields));

    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["m00".to_string()],
    );
    let mut aggr_exprs = (0..repeated_sums)
        .map(|index| AggregateExpr::new(AggregateFunction::Sum, Expr::column("m00"), format!("sum_m00_{index}")))
        .collect::<Vec<_>>();
    aggr_exprs.extend([
        AggregateExpr::new(AggregateFunction::Min, Expr::column("m00"), "min_m00"),
        AggregateExpr::new(AggregateFunction::Max, Expr::column("m00"), "max_m00"),
        AggregateExpr::new(AggregateFunction::Count, Expr::column("m00"), "count_m00"),
    ]);
    let aggregate = AggregateSpec::new(Vec::new(), aggr_exprs);

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let aggregate = builder.add_aggregate(source, aggregate, Arc::clone(&aggregated_schema));
    let plan = builder.finish_with_host_sink(aggregate);

    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 90_000,
            max_source_chunk_bytes: 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 3,
            min_retry_chunk_row_groups: 1,
        })
        .build()
        .unwrap();

    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.source_chunks_planned, 3);
    assert_eq!(metrics.source_chunks_committed, 3);
    assert_eq!(metrics.final_merge_attempts, 3);
    assert_eq!(metrics.final_merge_levels, 2);
    assert_eq!(metrics.retry_count, 1);
    assert_eq!(metrics.retry_reservation_failure_count, 1);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 1);
    assert_eq!(batch.num_columns(), repeated_sums + 3);
    assert!(batch.schema().fields().iter().all(|field| !field.name().starts_with("__cudf_global_valid_count_")));
    for index in 0..repeated_sums {
        assert_eq!(int64_array(batch, index).value(0), 3072);
    }
    assert_eq!(int64_array(batch, repeated_sums).value(0), 1);
    assert_eq!(int64_array(batch, repeated_sums + 1).value(0), 1);
    assert_eq!(int64_array(batch, repeated_sums + 2).value(0), 3072);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_reports_retry_facts_when_final_merge_retry_exhausts() {
    let dir = unique_temp_dir("cudf-query-engine-bounded-final-merge-retry-exhausted");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_final_merge_retry_input_parquet(&path, 1024);
    let aggregated_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("sum_m00", DataType::Int64, true),
    ]));

    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["key".to_string(), "m00".to_string()],
    );
    let aggregate = AggregateSpec::new(
        vec![Expr::column("key")],
        vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("m00"), "sum_m00")],
    );

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let aggregate = builder.add_aggregate(source, aggregate, aggregated_schema);
    let plan = builder.finish_with_host_sink(aggregate);

    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 50_000,
            max_source_chunk_bytes: 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 3,
            min_retry_chunk_row_groups: 1,
        })
        .build()
        .unwrap();

    let error = match execute_query_plan(&context, &plan) {
        Ok(_) => panic!("bounded execution should fail after exhausting final merge retry fan-in"),
        Err(error) => error,
    };

    assert_eq!(error.code(), QueryErrorCode::MemoryReservationFailed);
    assert_eq!(error.fact_value("memory_boundary"), Some("final_aggregate_merge"));
    assert_eq!(error.fact_value("reservation_kind"), Some("final_merge"));
    assert_eq!(error.fact_value("stage"), Some("final_merge"));
    assert_eq!(error.fact_value("attempt"), Some("2"));
    assert_eq!(error.fact_value("max_attempts"), Some("3"));
    assert_eq!(error.fact_value("retry_action"), Some("fail_min_merge_fan_in"));
    assert_eq!(error.fact_value("resume_boundary"), Some("final_merge"));
    assert_eq!(error.fact_value("committed_chunks"), Some("3"));
    assert_eq!(error.fact_value("committed_partials"), Some("3"));
    assert_eq!(error.fact_value("merge_level"), Some("1"));
    assert_eq!(error.fact_value("merge_fan_in"), Some("2"));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_retries_final_sort_with_hierarchical_topk() {
    let dir = unique_temp_dir("cudf-query-engine-bounded-final-sort-retry");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_single_row_group_final_sort_input_parquet(&path, 4096);
    let aggregated_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("sum_m00", DataType::Int64, true),
    ]));

    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["key".to_string(), "m00".to_string()],
    );
    let aggregate = AggregateSpec::new(
        vec![Expr::column("key")],
        vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("m00"), "sum_m00")],
    );
    let sort = SortSpec::new(vec![SortExpr::new(Expr::column("key"), true, false)]);
    let limit = LimitSpec::new(3, 7);

    let mut builder = QueryPlanBuilder::new();
    let source_id = builder.add_source(SourceNode::LocalParquet(source.clone()));
    let one_chunk_plan =
        plan_local_parquet_source_chunks(source_id, &source, SourceChunkPlanningPolicy::new(None, 1)).unwrap();
    assert_eq!(one_chunk_plan.chunk_count(), 1);
    let source_chunk_bytes = one_chunk_plan.chunks[0].known_row_group_bytes.unwrap_or(0);
    let aggregate = builder.add_aggregate(source_id, aggregate, Arc::clone(&aggregated_schema));
    let sort = builder.add_sort(aggregate, sort, Arc::clone(&aggregated_schema));
    let limit = builder.add_limit(sort, limit, Arc::clone(&aggregated_schema));
    let plan = builder.finish_with_host_sink(limit);

    let baseline_context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 8 * 1024 * 1024 * 1024,
            max_source_chunk_bytes: 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .build()
        .unwrap();
    let baseline = execute_query_plan(&baseline_context, &plan).unwrap();
    let QueryOutput::HostArrow(expected) = baseline.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    let resident_peak = baseline.metrics().resident_dataframe_peak_bytes.unwrap();
    let constrained_budget = source_chunk_bytes.saturating_add(resident_peak).saturating_add(4096);
    assert!(
        constrained_budget < resident_peak.saturating_mul(2),
        "test data must leave a budget window where source read fits but full final sort reservation fails"
    );

    let constrained_context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: constrained_budget,
            max_source_chunk_bytes: 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 3,
            min_retry_chunk_row_groups: 1,
        })
        .build()
        .unwrap();

    let result = execute_query_plan(&constrained_context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.source_chunks_planned, 1);
    assert_eq!(metrics.source_chunks_committed, 1);
    assert!(metrics.retry_count >= 1);
    assert!(metrics.retry_reservation_failure_count >= 1);
    assert!(metrics.reservation_denied_count >= 1);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), expected.num_rows());
    assert_eq!(int32_array(batch, 0).values().as_ref(), int32_array(expected, 0).values().as_ref());
    assert_eq!(int64_array(batch, 1).values().as_ref(), int64_array(expected, 1).values().as_ref());

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_reports_retry_facts_when_final_sort_without_limit_fails() {
    let dir = unique_temp_dir("cudf-query-engine-bounded-final-sort-retry-exhausted");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_single_row_group_final_sort_input_parquet(&path, 4096);
    let aggregated_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("sum_m00", DataType::Int64, true),
    ]));

    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["key".to_string(), "m00".to_string()],
    );
    let aggregate = AggregateSpec::new(
        vec![Expr::column("key")],
        vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("m00"), "sum_m00")],
    );
    let sort = SortSpec::new(vec![SortExpr::new(Expr::column("key"), true, false)]);

    let mut builder = QueryPlanBuilder::new();
    let source_id = builder.add_source(SourceNode::LocalParquet(source.clone()));
    let one_chunk_plan =
        plan_local_parquet_source_chunks(source_id, &source, SourceChunkPlanningPolicy::new(None, 1)).unwrap();
    assert_eq!(one_chunk_plan.chunk_count(), 1);
    let source_chunk_bytes = one_chunk_plan.chunks[0].known_row_group_bytes.unwrap_or(0);
    let aggregate = builder.add_aggregate(source_id, aggregate, Arc::clone(&aggregated_schema));
    let sort = builder.add_sort(aggregate, sort, aggregated_schema);
    let plan = builder.finish_with_host_sink(sort);

    let baseline_context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 8 * 1024 * 1024 * 1024,
            max_source_chunk_bytes: 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .build()
        .unwrap();
    let baseline = execute_query_plan(&baseline_context, &plan).unwrap();
    let resident_peak = baseline.metrics().resident_dataframe_peak_bytes.unwrap();
    let constrained_budget = source_chunk_bytes.saturating_add(resident_peak).saturating_add(4096);

    let constrained_context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: constrained_budget,
            max_source_chunk_bytes: 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 3,
            min_retry_chunk_row_groups: 1,
        })
        .build()
        .unwrap();

    let error = match execute_query_plan(&constrained_context, &plan) {
        Ok(_) => panic!("bounded execution should fail final sort when no top-k retry boundary exists"),
        Err(error) => error,
    };

    assert_eq!(error.code(), QueryErrorCode::MemoryReservationFailed);
    assert_eq!(error.fact_value("memory_boundary"), Some("final_sort"));
    assert_eq!(error.fact_value("reservation_kind"), Some("final_sort"));
    assert_eq!(error.fact_value("stage"), Some("final_sort"));
    assert_eq!(error.fact_value("attempt"), Some("1"));
    assert_eq!(error.fact_value("max_attempts"), Some("3"));
    assert_eq!(error.fact_value("retry_action"), Some("retry_attempts_exhausted"));
    assert_eq!(error.fact_value("resume_boundary"), Some("final_sort"));
    assert_eq!(error.fact_value("committed_chunks"), Some("1"));
    assert_eq!(error.fact_value("committed_partials"), Some("1"));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_executes_non_aggregate_source_plan() {
    let dir = unique_temp_dir("cudf-query-engine-bounded-source-only");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let schema = write_row_grouped_input_parquet(&path);
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&schema),
        vec!["key".into(), "payload".into(), "m00".into(), "m01".into()],
    );

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let plan = builder.finish_with_host_sink(source);
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
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should return a RecordBatch");
    };
    assert_eq!(batch.num_rows(), 6);
    assert_eq!(result.metrics().parquet_files, 1);
    assert!(result.metrics().reservation_granted_bytes > 0);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_rejects_unsupported_expression_before_source_read() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("value", DataType::Int64, false),
    ]));
    let aggregated_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("sum_value", DataType::Int64, true),
    ]));
    let source = LocalParquetSource::new(
        "/tmp/does-not-need-to-exist-for-expression-preflight.parquet",
        Arc::clone(&input_schema),
        vec!["key".into(), "value".into()],
    );
    let unsupported_filter =
        Expr::binary(Expr::column("key"), BinaryOp::Eq, Expr::literal(LiteralValue::Null(DataType::Binary)));
    let aggregate = AggregateSpec::new(
        vec![Expr::column("key")],
        vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("value"), "sum_value")],
    );

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let filter = builder.add_filter(source, unsupported_filter, Arc::clone(&input_schema));
    let aggregate = builder.add_aggregate(filter, aggregate, aggregated_schema);
    let plan = builder.finish_with_host_sink(aggregate);
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

    let error = match execute_query_plan(&context, &plan) {
        Ok(_) => panic!("bounded execution should reject unsupported expressions before source read"),
        Err(error) => error,
    };

    assert_eq!(error.code(), QueryErrorCode::UnsupportedType);
    assert_eq!(error.operation(), "cudf_query_engine.plan.validate_expr");
    assert!(error.fact_value("path").is_none());
}

#[test]
fn bounded_native_query_executes_root_hash_join_with_ordered_reservations() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let dir = unique_temp_dir("cudf-query-engine-bounded-root-hash-join");
    let left_path = dir.join("lineitem.parquet");
    let right_path = dir.join("orders.parquet");
    let left_schema = write_join_left_parquet(&runtime, &left_path);
    let right_schema = write_join_right_parquet(&runtime, &right_path);
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
        Field::new("category", DataType::Utf8, false),
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
        JoinSpec::new(JoinType::Inner, vec![join_order_id_key()])
            .with_output_projection(vec![
                JoinOutputColumn::left(cudf_query_engine::ColumnRef::input(0, "order_id", "order_id"), "order_id"),
                JoinOutputColumn::left(cudf_query_engine::ColumnRef::input(0, "qty", "qty"), "qty"),
                JoinOutputColumn::right(cudf_query_engine::ColumnRef::input(1, "category", "category"), "category"),
            ])
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
    assert_eq!(metrics.parquet_files, 2);
    assert_eq!(metrics.reservation_denied_count, 0);
    assert!(metrics.reservation_requested_bytes > 0);
    assert!(metrics.reservation_granted_bytes > 0);
    assert_eq!(metrics.hash_join_build_count, 1);
    assert!(metrics.hash_join_hash_table_bytes > 0);
    let join_metrics =
        metrics.operators.iter().find_map(|operator| operator.join.as_ref()).expect("join metrics should be recorded");
    assert_eq!(join_metrics.join_type, "inner");
    assert_eq!(join_metrics.build_side, "right");
    assert_eq!(join_metrics.probe_side, "left");
    assert!(join_metrics.gather_map_bytes > 0);
    assert_eq!(join_metrics.output_rows, 3);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 3);
    assert_eq!(batch.num_columns(), 3);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[2, 2, 3]);
    assert_eq!(int32_array(batch, 1).values().as_ref(), &[20, 25, 30]);
    assert_eq!(string_array(batch, 2).iter().collect::<Vec<_>>(), vec![Some("two"), Some("two"), Some("three")]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_executes_root_keyless_conditional_join_with_ordered_reservations() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let dir = unique_temp_dir("cudf-query-engine-bounded-root-conditional-join");
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
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(JoinType::Inner, vec![])
            .with_residual_predicate(qty_gt_right_order_id_times_ten())
            .with_output_projection(vec![
                JoinOutputColumn::left(cudf_query_engine::ColumnRef::input(0, "qty", "qty"), "qty"),
                JoinOutputColumn::right(cudf_query_engine::ColumnRef::input(1, "category", "category"), "category"),
            ]),
        Arc::clone(&output_schema),
    );
    let plan = builder.finish_with_host_sink(join);
    assert!(plan.has_bounded_executable_root_join());

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
    assert_eq!(metrics.parquet_files, 2);
    assert_eq!(metrics.reservation_denied_count, 0);
    assert!(metrics.reservation_requested_bytes > 0);
    assert!(metrics.reservation_granted_bytes > 0);
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
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let dir = unique_temp_dir("cudf-query-engine-bounded-root-conditional-left-join");
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
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(JoinType::Left, vec![])
            .with_residual_predicate(qty_gt_right_order_id_times_ten())
            .with_output_projection(vec![
                JoinOutputColumn::left(cudf_query_engine::ColumnRef::input(0, "qty", "qty"), "qty"),
                JoinOutputColumn::right(cudf_query_engine::ColumnRef::input(1, "category", "category"), "category"),
            ])
            .with_build_side(JoinSide::Right),
        Arc::clone(&output_schema),
    );
    let plan = builder.finish_with_host_sink(join);
    assert!(plan.has_bounded_executable_root_join());

    let context = bounded_join_execution_context(&runtime);
    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.phase_c_memory_control_policy.as_str(), "bounded");
    assert_eq!(metrics.reservation_denied_count, 0);
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
fn bounded_native_query_executes_root_keyless_conditional_left_only_joins_with_left_build_orientation() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let dir = unique_temp_dir("cudf-query-engine-bounded-root-conditional-left-only-joins");
    let left_path = dir.join("lineitem.parquet");
    let right_path = dir.join("orders.parquet");
    let left_schema = write_join_left_parquet(&runtime, &left_path);
    let right_schema = write_join_right_parquet(&runtime, &right_path);

    let semi = execute_bounded_keyless_left_only_conditional_join(
        &runtime,
        &left_path,
        &right_path,
        &left_schema,
        &right_schema,
        JoinType::LeftSemi,
    );
    let anti = execute_bounded_keyless_left_only_conditional_join(
        &runtime,
        &left_path,
        &right_path,
        &left_schema,
        &right_schema,
        JoinType::LeftAnti,
    );

    let QueryOutput::HostArrow(semi_batch) = semi.output() else {
        panic!("left semi bounded conditional join should return host output");
    };
    let mut semi_rows = int32_array(semi_batch, 0).values().iter().copied().collect::<Vec<_>>();
    semi_rows.sort();
    assert_eq!(semi_rows, vec![25, 30]);
    let semi_metrics = semi
        .metrics()
        .operators
        .iter()
        .find_map(|operator| operator.join.as_ref())
        .expect("join metrics should be recorded");
    assert_eq!(semi_metrics.join_type, "left_semi");
    assert_eq!(semi_metrics.execution_strategy, "bounded_keyless_conditional_join");
    assert_eq!(semi_metrics.build_side, "left");
    assert_eq!(semi_metrics.key_count, 0);
    assert_eq!(semi_metrics.output_rows, 2);
    assert!(semi_metrics.gather_map_bytes > 0);

    let QueryOutput::HostArrow(anti_batch) = anti.output() else {
        panic!("left anti bounded conditional join should return host output");
    };
    let mut anti_rows = int32_array(anti_batch, 0).values().iter().copied().collect::<Vec<_>>();
    anti_rows.sort();
    assert_eq!(anti_rows, vec![10, 20]);
    let anti_metrics = anti
        .metrics()
        .operators
        .iter()
        .find_map(|operator| operator.join.as_ref())
        .expect("join metrics should be recorded");
    assert_eq!(anti_metrics.join_type, "left_anti");
    assert_eq!(anti_metrics.execution_strategy, "bounded_keyless_conditional_join");
    assert_eq!(anti_metrics.build_side, "left");
    assert_eq!(anti_metrics.key_count, 0);
    assert_eq!(anti_metrics.output_rows, 2);
    assert!(anti_metrics.gather_map_bytes > 0);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_executes_root_left_hash_join_with_unmatched_probe_rows() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let dir = unique_temp_dir("cudf-query-engine-bounded-root-left-hash-join");
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
                JoinOutputColumn::left(cudf_query_engine::ColumnRef::input(0, "order_id", "order_id"), "order_id"),
                JoinOutputColumn::left(cudf_query_engine::ColumnRef::input(0, "qty", "qty"), "qty"),
                JoinOutputColumn::right(cudf_query_engine::ColumnRef::input(1, "category", "category"), "category"),
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
    let dir = unique_temp_dir("cudf-query-engine-bounded-root-full-hash-join");
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
                    cudf_query_engine::ColumnRef::input(0, "order_id", "left_order_id"),
                    "left_order_id",
                ),
                JoinOutputColumn::right(
                    cudf_query_engine::ColumnRef::input(1, "order_id", "right_order_id"),
                    "right_order_id",
                ),
                JoinOutputColumn::left(cudf_query_engine::ColumnRef::input(0, "qty", "qty"), "qty"),
                JoinOutputColumn::right(cudf_query_engine::ColumnRef::input(1, "category", "category"), "category"),
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
    let mut rows: Vec<(Option<i32>, Option<i32>, Option<i32>, Option<String>)> = (0..batch.num_rows())
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

#[test]
fn bounded_native_query_executes_root_hash_join_under_aggregate_with_ordered_reservations() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let dir = unique_temp_dir("cudf-query-engine-bounded-root-join-aggregate");
    let left_path = dir.join("lineitem.parquet");
    let right_path = dir.join("orders.parquet");
    let left_schema = write_join_left_parquet(&runtime, &left_path);
    let right_schema = write_join_right_parquet(&runtime, &right_path);
    let join_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
    ]));
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("line_count", DataType::Int64, false),
    ]));
    let sort = SortSpec::new(vec![SortExpr::new(Expr::column("order_id"), false, false)]);

    let mut builder = QueryPlanBuilder::new();
    let left =
        builder.add_source(SourceNode::LocalParquet(join_left_source(&left_path, &left_schema, &["order_id", "qty"])));
    let right =
        builder.add_source(SourceNode::LocalParquet(join_right_source(&right_path, &right_schema, &["order_id"])));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(JoinType::Inner, vec![join_order_id_key()])
            .with_output_projection(vec![
                JoinOutputColumn::left(cudf_query_engine::ColumnRef::input(0, "order_id", "order_id"), "order_id"),
                JoinOutputColumn::left(cudf_query_engine::ColumnRef::input(0, "qty", "qty"), "qty"),
            ])
            .with_build_side(JoinSide::Right),
        join_schema,
    );
    let aggregate = builder.add_aggregate(
        join,
        AggregateSpec::new(
            vec![Expr::column("order_id")],
            vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("qty"), "line_count")],
        ),
        Arc::clone(&aggregate_schema),
    );
    let sort = builder.add_sort(aggregate, sort, Arc::clone(&aggregate_schema));
    let limit = builder.add_limit(sort, LimitSpec::fetch(10), aggregate_schema);
    let plan = builder.finish_with_host_sink(limit);
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
    assert!(metrics.reservation_requested_bytes > 0);
    assert!(metrics.reservation_granted_bytes > 0);
    assert_eq!(metrics.hash_join_build_count, 1);
    assert_eq!(metrics.partial_aggregate_count, 0);
    assert_eq!(metrics.final_merge_attempts, 0);
    assert_eq!(metrics.boundary.device_to_host_materializations, 1);
    let join_metrics =
        metrics.operators.iter().find_map(|operator| operator.join.as_ref()).expect("join metrics should be recorded");
    assert_eq!(join_metrics.join_type, "inner");
    assert_eq!(join_metrics.output_rows, 3);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 2);
    assert_eq!(batch.num_columns(), 2);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[2, 3]);
    assert_eq!(int64_array(batch, 1).values().as_ref(), &[2, 1]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_rejects_unsupported_join_under_aggregate_before_source_read() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let left_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("amount", DataType::Int64, false),
    ]));
    let right_schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    let join_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("amount", DataType::Int64, false),
    ]));
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("sum_amount", DataType::Int64, true),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/bounded-aggregate-join-left-does-not-need-to-exist.parquet",
        Arc::clone(&left_schema),
        vec!["id".into(), "amount".into()],
    )));
    let right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/bounded-aggregate-join-right-does-not-need-to-exist.parquet",
        Arc::clone(&right_schema),
        vec!["id".into()],
    )));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(
            JoinType::LeftSemi,
            vec![JoinKey::new(
                cudf_query_engine::ColumnRef::input(0, "id", "left_id"),
                cudf_query_engine::ColumnRef::input(1, "id", "right_id"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(cudf_query_engine::ColumnRef::input(0, "id", "id"), "id"),
            JoinOutputColumn::left(cudf_query_engine::ColumnRef::input(0, "amount", "amount"), "amount"),
        ])
        .with_build_side(JoinSide::Right),
        join_schema,
    );
    let aggregate = builder.add_aggregate(
        join,
        AggregateSpec::new(
            vec![Expr::column("id")],
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("amount"), "sum_amount")],
        ),
        aggregate_schema,
    );
    let plan = builder.finish_with_host_sink(aggregate);
    assert!(!plan.has_bounded_executable_root_join());
    assert!(!plan.has_bounded_executable_join_pipeline());

    let context = ExecutionContext::builder()
        .with_shared_runtime(runtime)
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 1024 * 1024,
            max_source_chunk_bytes: 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .build()
        .unwrap();

    let error = match execute_query_plan(&context, &plan) {
        Ok(_) => panic!("bounded aggregate join should reject before source reads"),
        Err(error) => error,
    };

    assert_bounded_join_rejection_contract(&error);
}

fn assert_bounded_join_rejection_contract(error: &cudf_query_engine::QueryError) {
    let contract = BoundedJoinReservationContract;

    assert_eq!(error.code(), QueryErrorCode::UnsupportedPlanShape);
    assert_eq!(error.operation(), "cudf_query_engine.bounded.plan");
    assert_eq!(error.fact_value("reason"), Some("bounded_join_plan_shape_unsupported"));
    assert_eq!(error.fact_value("bounded_join_reservation_model"), Some(contract.model_status()));
    assert_eq!(error.fact_value("bounded_join_execution_status"), Some(contract.execution_status()));
    assert_eq!(error.fact_value("bounded_join_execution_families"), Some(contract.execution_families_csv().as_str()));
    assert_eq!(error.fact_value("bounded_join_enabled"), Some("true"));
    assert_eq!(
        error.fact_value("boundary_policy"),
        Some(
            "source_read=direct_device_local_parquet,build_materialization=query_scoped_gpu_hash_join,probe_execution=reusable_gpu_hash_join,result_materialization=explicit_sink"
        )
    );
    assert_eq!(error.fact_value("source_read_boundary"), Some("direct_device_local_parquet"));
    assert_eq!(error.fact_value("build_materialization_boundary"), Some("query_scoped_gpu_hash_join"));
    assert_eq!(error.fact_value("probe_execution_boundary"), Some("reusable_gpu_hash_join"));
    assert_eq!(error.fact_value("result_materialization_boundary"), Some("explicit_sink"));
    assert_eq!(error.fact_value("bounded_join_reservation_kinds"), Some(contract.reservation_kinds_csv().as_str()));
    assert_eq!(
        error.fact_value("bounded_join_reservation_statuses"),
        Some(contract.reservation_statuses_csv().as_str())
    );
    assert_eq!(error.fact_value("blocked_reservations"), Some(contract.blocking_reservation_kinds_csv().as_str()));
    assert_eq!(error.fact_value("modeled_reservations"), Some(contract.modeled_reservation_kinds_csv().as_str()));
    assert_eq!(error.fact_value("deferred_reservations"), Some(contract.deferred_reservation_kinds_csv().as_str()));
    assert_eq!(
        error.fact_value("bounded_join_lower_level_blockers"),
        Some(contract.lower_level_blockers_csv().as_str())
    );
    assert_eq!(
        error.fact_value("bounded_join_reservation_requirements"),
        Some(contract.requirement_details_csv().as_str())
    );
    assert_eq!(error.fact_value("cudf_rs_hash_table_observability"), Some("hash_table_device_allocated_size_bytes"));
    assert_eq!(error.fact_value("cudf_rs_gather_map_observability"), Some("join_indices_device_allocated_size_bytes"));
    assert_eq!(
        error.fact_value("cudf_rs_hash_join_gather_map_reservation_contract"),
        Some("hash_join_gather_map_reservation_size_bytes")
    );
    assert_eq!(error.fact_value("cudf_rs_hash_join_output_size_observability"), Some("hash_join_exact_output_size"));
    assert_eq!(error.fact_value("cudf_rs_hash_join_probe_output_size_hint"), Some("hash_join_probe_with_output_size"));
    assert_eq!(
        error.fact_value("cudf_rs_conditional_join_output_size_observability"),
        Some("conditional_join_exact_output_size")
    );
    assert_eq!(
        error.fact_value("cudf_rs_conditional_join_output_size_hint"),
        Some("conditional_join_indices_with_output_size")
    );
    assert_eq!(
        error.fact_value("cudf_rs_hash_join_pre_build_reservation_contract"),
        Some("hash_join_pre_build_reservation_size_bytes")
    );
    assert_eq!(
        error.fact_value("cudf_rs_hash_join_pre_build_reservation_status"),
        Some("available_exact_flat_hash_join_builds")
    );
    assert_eq!(
        error.fact_value("cudf_rs_dataframe_gather_output_reservation_contract"),
        Some("dataframe_gather_output_reservation_size_bytes")
    );
    assert_eq!(error.fact_value("missing_lower_level_contract"), None);
    assert_eq!(error.fact_value("missing_reservations"), Some(contract.blocking_reservation_kinds_csv().as_str()));
    assert!(error.facts().iter().all(|(key, value)| !key.contains("estimate") && !value.contains("estimate")));
    assert_eq!(error.fact_value("path"), None);
    assert_eq!(error.fact_value("file_index"), None);
    assert_eq!(error.fact_value("row_groups"), None);
    assert_eq!(error.fact_value("chunk_index"), None);
    assert!(error.source_error().is_none());
}

fn join_left_source(path: &Path, schema: &Arc<Schema>, columns: &[&str]) -> LocalParquetSource {
    LocalParquetSource::from_files(
        vec![LocalParquetFile::new(path).with_file_size_bytes(fs::metadata(path).unwrap().len())],
        Arc::clone(schema),
        columns.iter().map(|column| (*column).to_string()).collect(),
    )
}

fn join_right_source(path: &Path, schema: &Arc<Schema>, columns: &[&str]) -> LocalParquetSource {
    LocalParquetSource::from_files(
        vec![LocalParquetFile::new(path).with_file_size_bytes(fs::metadata(path).unwrap().len())],
        Arc::clone(schema),
        columns.iter().map(|column| (*column).to_string()).collect(),
    )
}

fn join_order_id_key() -> JoinKey {
    JoinKey::new(
        cudf_query_engine::ColumnRef::input(0, "order_id", "left_order_id"),
        cudf_query_engine::ColumnRef::input(1, "order_id", "right_order_id"),
    )
}

fn id_join_key() -> JoinKey {
    JoinKey::new(
        cudf_query_engine::ColumnRef::input(0, "id", "left_id"),
        cudf_query_engine::ColumnRef::input(1, "id", "right_id"),
    )
}

fn right_order_id_times_ten() -> Expr {
    Expr::binary(
        Expr::input_column(1, "order_id", "right_order_id"),
        BinaryOp::Multiply,
        Expr::literal(LiteralValue::Int32(10)),
    )
}

fn qty_gt_right_order_id_times_ten() -> Expr {
    Expr::binary(Expr::input_column(0, "qty", "qty"), BinaryOp::Gt, right_order_id_times_ten())
}

fn bounded_join_execution_context(runtime: &Arc<cudf::Runtime>) -> ExecutionContext {
    ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 1024 * 1024,
            max_source_chunk_bytes: 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .build()
        .unwrap()
}

fn execute_bounded_keyless_left_only_conditional_join(
    runtime: &Arc<cudf::Runtime>,
    left_path: &Path,
    right_path: &Path,
    left_schema: &Arc<Schema>,
    right_schema: &Arc<Schema>,
    join_type: JoinType,
) -> cudf_query_engine::QueryExecutionResult {
    let output_schema = Arc::new(Schema::new(vec![Field::new("qty", DataType::Int32, false)]));

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(join_left_source(left_path, left_schema, &["qty"])));
    let right =
        builder.add_source(SourceNode::LocalParquet(join_right_source(right_path, right_schema, &["order_id"])));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(join_type, vec![])
            .with_residual_predicate(qty_gt_right_order_id_times_ten())
            .with_output_projection(vec![JoinOutputColumn::left(
                cudf_query_engine::ColumnRef::input(0, "qty", "qty"),
                "qty",
            )])
            .with_build_side(JoinSide::Left),
        Arc::clone(&output_schema),
    );
    let plan = builder.finish_with_host_sink(join);
    assert!(plan.has_bounded_executable_root_join());
    let context = bounded_join_execution_context(runtime);
    execute_query_plan(&context, &plan).unwrap()
}

fn execute_left_only_residual_join(
    runtime: &Arc<cudf::Runtime>,
    left_path: &Path,
    right_path: &Path,
    left_schema: &Arc<Schema>,
    right_schema: &Arc<Schema>,
    join_type: JoinType,
) -> cudf_query_engine::QueryExecutionResult {
    let output_schema = Arc::new(Schema::new(vec![Field::new("qty", DataType::Int32, false)]));

    let mut builder = QueryPlanBuilder::new();
    let left =
        builder.add_source(SourceNode::LocalParquet(join_left_source(left_path, left_schema, &["order_id", "qty"])));
    let right =
        builder.add_source(SourceNode::LocalParquet(join_right_source(right_path, right_schema, &["order_id"])));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(join_type, vec![join_order_id_key()])
            .with_residual_predicate(qty_gt_right_order_id_times_ten())
            .with_output_projection(vec![JoinOutputColumn::left(
                cudf_query_engine::ColumnRef::input(0, "qty", "qty"),
                "qty",
            )])
            .with_build_side(JoinSide::Right),
        Arc::clone(&output_schema),
    );
    let sort = builder.add_sort(
        join,
        SortSpec::new(vec![SortExpr::new(Expr::column("qty"), false, false)]),
        Arc::clone(&output_schema),
    );
    let plan = builder.finish_with_host_sink(sort);
    let context = ExecutionContext::from_shared_runtime(Arc::clone(runtime));
    execute_query_plan(&context, &plan).unwrap()
}

fn residual_validation_error(join_type: JoinType, keys: Vec<JoinKey>, residual: Expr) -> cudf_query_engine::QueryError {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("left_id", DataType::Int32, true),
        Field::new("right_id", DataType::Int32, true),
    ]));
    let left_source = LocalParquetSource::new(
        "/tmp/does-not-need-to-exist-for-residual-validation-left.parquet",
        Arc::clone(&schema),
        vec!["id".into()],
    );
    let right_source = LocalParquetSource::new(
        "/tmp/does-not-need-to-exist-for-residual-validation-right.parquet",
        Arc::clone(&schema),
        vec!["id".into()],
    );

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(left_source));
    let right = builder.add_source(SourceNode::LocalParquet(right_source));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(join_type, keys).with_residual_predicate(residual).with_output_projection(vec![
            JoinOutputColumn::left(cudf_query_engine::ColumnRef::input(0, "id", "left_id"), "left_id"),
            JoinOutputColumn::right(cudf_query_engine::ColumnRef::input(1, "id", "right_id"), "right_id"),
        ]),
        output_schema,
    );
    let plan = builder.finish_with_host_sink(join);
    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime));

    match execute_query_plan(&context, &plan) {
        Ok(_) => panic!("native residual validation should reject invalid predicates before source read"),
        Err(error) => error,
    }
}

fn write_input_parquet(runtime: &cudf::Runtime, path: &std::path::Path) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int64, false),
        Field::new("m00", DataType::Int64, false),
        Field::new("m01", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 1, 3])),
            Arc::new(Int64Array::from(vec![5, 10, 20, 20])),
            Arc::new(Int64Array::from(vec![100, 200, 300, 400])),
            Arc::new(Int64Array::from(vec![1, 2, 3, 4])),
        ],
    )
    .unwrap();
    DataFrame::from_record_batch(runtime, &batch).unwrap().write_parquet(path).unwrap();
    schema
}

fn write_join_left_parquet(runtime: &cudf::Runtime, path: &std::path::Path) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(vec![1, 2, 2, 3])), Arc::new(Int32Array::from(vec![10, 20, 25, 30]))],
    )
    .unwrap();
    DataFrame::from_record_batch(runtime, &batch).unwrap().write_parquet(path).unwrap();
    schema
}

fn write_join_left_decimal_parquet(runtime: &cudf::Runtime, path: &std::path::Path) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("amount", DataType::Decimal128(15, 2), false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 2, 3])),
            Arc::new(
                Decimal128Array::from_iter_values([12_345, 20_000, 999, 30_000])
                    .with_precision_and_scale(15, 2)
                    .unwrap(),
            ),
        ],
    )
    .unwrap();
    DataFrame::from_record_batch(runtime, &batch).unwrap().write_parquet(path).unwrap();
    schema
}

fn write_join_right_parquet(runtime: &cudf::Runtime, path: &std::path::Path) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("category", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(vec![2, 3, 4])), Arc::new(StringArray::from(vec!["two", "three", "four"]))],
    )
    .unwrap();
    DataFrame::from_record_batch(runtime, &batch).unwrap().write_parquet(path).unwrap();
    schema
}

fn write_tpch_like_types_parquet(runtime: &cudf::Runtime, path: &std::path::Path) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("orderkey", DataType::Int64, false),
        Field::new("amount", DataType::Decimal128(15, 2), false),
        Field::new("shipdate", DataType::Date32, false),
        Field::new("event_ts", DataType::Timestamp(TimeUnit::Nanosecond, None), false),
        Field::new("status", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 3])),
            Arc::new(
                Decimal128Array::from_iter_values([12_345, 67_890, 22_222]).with_precision_and_scale(15, 2).unwrap(),
            ),
            Arc::new(Date32Array::from(vec![18_262, 18_293, 18_322])),
            Arc::new(TimestampNanosecondArray::from(vec![100, 200, 300])),
            Arc::new(StringArray::from(vec!["O", "F", "O"])),
        ],
    )
    .unwrap();
    DataFrame::from_record_batch(runtime, &batch).unwrap().write_parquet(path).unwrap();
    schema
}

fn write_q6_lineitem_parquet(path: &Path) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("l_quantity", DataType::Decimal128(15, 2), false),
        Field::new("l_extendedprice", DataType::Decimal128(15, 2), false),
        Field::new("l_discount", DataType::Decimal128(15, 2), false),
        Field::new("l_shipdate", DataType::Date32, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(
                Decimal128Array::from_iter_values([2_300, 2_500, 1_000, 1_200, 1_200, 1_200, 1_200])
                    .with_precision_and_scale(15, 2)
                    .unwrap(),
            ),
            Arc::new(
                Decimal128Array::from_iter_values([10_000, 20_000, 30_000, 40_000, 50_000, 60_000, 70_000])
                    .with_precision_and_scale(15, 2)
                    .unwrap(),
            ),
            Arc::new(Decimal128Array::from_iter_values([5, 6, 7, 6, 4, 8, 6]).with_precision_and_scale(15, 2).unwrap()),
            Arc::new(Date32Array::from(vec![
                date32(1994, 1, 15),
                date32(1994, 6, 1),
                date32(1994, 9, 30),
                date32(1995, 1, 1),
                date32(1994, 3, 15),
                date32(1994, 11, 15),
                date32(1993, 12, 31),
            ])),
        ],
    )
    .unwrap();
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(2)).build();
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    schema
}

fn write_phase2_scalar_expression_parquet(runtime: &cudf::Runtime, path: &std::path::Path) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("orderkey", DataType::Int64, false),
        Field::new("shipdate", DataType::Date32, false),
        Field::new("event_ts", DataType::Timestamp(TimeUnit::Nanosecond, None), false),
        Field::new("status", DataType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 3, 4])),
            Arc::new(Date32Array::from(vec![18_262, 18_293, 18_322, 18_336])),
            Arc::new(TimestampNanosecondArray::from(vec![100, 200, 300, 400])),
            Arc::new(StringArray::from(vec![Some("O"), Some("F"), Some("O"), None])),
        ],
    )
    .unwrap();
    DataFrame::from_record_batch(runtime, &batch).unwrap().write_parquet(path).unwrap();
    schema
}

fn partition_column_query_plan(
    first_path: &Path,
    second_path: &Path,
    file_schema: Arc<Schema>,
) -> cudf_query_engine::QueryPlan {
    let source_schema = Arc::new(Schema::new(
        file_schema
            .fields()
            .iter()
            .map(|field| field.as_ref().clone())
            .chain([Field::new("part", DataType::Int32, false)])
            .collect::<Vec<_>>(),
    ));
    let projected_schema = Arc::new(Schema::new(vec![
        Field::new("part", DataType::Int32, false),
        Field::new("m00", DataType::Int64, false),
    ]));
    let aggregated_schema = Arc::new(Schema::new(vec![
        Field::new("part", DataType::Int32, false),
        Field::new("sum_m00", DataType::Int64, true),
    ]));
    let source = LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new(first_path)
                .with_file_size_bytes(fs::metadata(first_path).unwrap().len())
                .with_partition_values(vec![LiteralValue::Int32(1)]),
            LocalParquetFile::new(second_path)
                .with_file_size_bytes(fs::metadata(second_path).unwrap().len())
                .with_partition_values(vec![LiteralValue::Int32(2)]),
        ],
        Arc::clone(&source_schema),
        vec!["m00".to_string()],
    )
    .with_partition_columns(vec![PartitionColumn::new("part", DataType::Int32, false)])
    .with_column_roles(vec!["part".to_string(), "m00".to_string()], vec!["part".to_string()]);
    let filter = Expr::binary(Expr::column("part"), BinaryOp::Eq, Expr::literal(LiteralValue::Int32(2)));
    let projection = vec![Expr::column("part"), Expr::column("m00")];
    let aggregate = AggregateSpec::new(
        vec![Expr::column("part")],
        vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("m00"), "sum_m00")],
    );
    let sort = SortSpec::new(vec![SortExpr::new(Expr::column("part"), false, false)]);

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let filter = builder.add_filter(source, filter, source_schema);
    let projection = builder.add_projection(filter, projection, projected_schema);
    let aggregate = builder.add_aggregate(projection, aggregate, Arc::clone(&aggregated_schema));
    let sort = builder.add_sort(aggregate, sort, Arc::clone(&aggregated_schema));
    builder.finish_with_host_sink(sort)
}

fn write_row_grouped_input_parquet(path: &Path) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int64, false),
        Field::new("m00", DataType::Int64, false),
        Field::new("m01", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 1, 3, 2, 3])),
            Arc::new(Int64Array::from(vec![10, 10, 20, 20, 30, 5])),
            Arc::new(Int64Array::from(vec![100, 200, 300, 400, 500, 600])),
            Arc::new(Int64Array::from(vec![1, 2, 3, 4, 5, 6])),
        ],
    )
    .unwrap();
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(2)).build();
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    schema
}

fn write_uneven_avg_input_parquet(path: &Path) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("payload", DataType::Int64, false),
        Field::new("m00", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 2, 1, 1, 3])),
            Arc::new(Int64Array::from(vec![10, 10, 10, 10, 10, 10])),
            Arc::new(Int64Array::from(vec![0, 100, 200, 30, 30, 300])),
        ],
    )
    .unwrap();
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(3)).build();
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    schema
}

fn write_uneven_global_avg_input_parquet(path: &Path) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![Field::new("m00", DataType::Int64, false)]));
    let batch =
        RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int64Array::from(vec![0, 100, 200, 30, 30]))]).unwrap();
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(4)).build();
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    schema
}

fn write_decimal_avg_input_parquet(path: &Path) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![Field::new("amount", DataType::Decimal128(15, 2), false)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Decimal128Array::from_iter_values([100, 200, 300]).with_precision_and_scale(15, 2).unwrap())],
    )
    .unwrap();
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(1)).build();
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    schema
}

fn write_nullable_avg_input_parquet(path: &Path) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("m00", DataType::Int64, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 1, 2, 2, 3])),
            Arc::new(Int64Array::from(vec![None, None, Some(10), None, Some(30)])),
        ],
    )
    .unwrap();
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(2)).build();
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    schema
}

fn write_nullable_global_aggregate_input_parquet(path: &Path) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("payload", DataType::Int64, false),
        Field::new("m00", DataType::Int64, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int64Array::from(vec![1, 2, 3, 4])), Arc::new(Int64Array::from(vec![None, None, None, None]))],
    )
    .unwrap();
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(2)).build();
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    schema
}

fn nullable_global_aggregate_plan(
    path: &Path,
    input_schema: Arc<Schema>,
    filter: Option<Expr>,
) -> cudf_query_engine::QueryPlan {
    let aggregated_schema = Arc::new(Schema::new(vec![
        Field::new("sum_m00", DataType::Int64, true),
        Field::new("avg_m00", DataType::Float64, true),
        Field::new("min_m00", DataType::Int64, true),
        Field::new("max_m00", DataType::Int64, true),
        Field::new("count_m00", DataType::Int64, false),
        Field::new("count_rows", DataType::Int64, false),
    ]));
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(path).with_file_size_bytes(fs::metadata(path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["payload".to_string(), "m00".to_string()],
    );
    let aggregate = AggregateSpec::new(
        Vec::new(),
        vec![
            AggregateExpr::new(AggregateFunction::Sum, Expr::column("m00"), "sum_m00"),
            AggregateExpr::new(AggregateFunction::Avg, Expr::column("m00"), "avg_m00"),
            AggregateExpr::new(AggregateFunction::Min, Expr::column("m00"), "min_m00"),
            AggregateExpr::new(AggregateFunction::Max, Expr::column("m00"), "max_m00"),
            AggregateExpr::new(AggregateFunction::Count, Expr::column("m00"), "count_m00"),
            AggregateExpr::new(AggregateFunction::Count, Expr::literal(LiteralValue::Int64(1)), "count_rows"),
        ],
    );

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let aggregate_input =
        if let Some(filter) = filter { builder.add_filter(source, filter, input_schema) } else { source };
    let aggregate = builder.add_aggregate(aggregate_input, aggregate, aggregated_schema);
    builder.finish_with_host_sink(aggregate)
}

fn write_final_merge_retry_input_parquet(path: &Path, keys_per_row_group: i32) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("m00", DataType::Int64, false),
    ]));
    let keys = (0..3).flat_map(|_| 0..keys_per_row_group).collect::<Vec<_>>();
    let values = vec![1_i64; keys.len()];
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(keys)), Arc::new(Int64Array::from(values))],
    )
    .unwrap();
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(keys_per_row_group as usize)).build();
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    schema
}

fn write_single_row_group_final_sort_input_parquet(path: &Path, rows: i32) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("m00", DataType::Int64, false),
    ]));
    let keys = (0..rows).collect::<Vec<_>>();
    let values = vec![1_i64; rows as usize];
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(keys)), Arc::new(Int64Array::from(values))],
    )
    .unwrap();
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(rows as usize)).build();
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    schema
}

fn int32_array(batch: &RecordBatch, index: usize) -> &Int32Array {
    batch.column(index).as_any().downcast_ref::<Int32Array>().expect("expected Int32Array")
}

fn int64_array(batch: &RecordBatch, index: usize) -> &Int64Array {
    batch.column(index).as_any().downcast_ref::<Int64Array>().expect("expected Int64Array")
}

fn boolean_array(batch: &RecordBatch, index: usize) -> &BooleanArray {
    batch.column(index).as_any().downcast_ref::<BooleanArray>().expect("expected BooleanArray")
}

fn float64_array(batch: &RecordBatch, index: usize) -> &Float64Array {
    batch.column(index).as_any().downcast_ref::<Float64Array>().expect("expected Float64Array")
}

fn decimal128_array(batch: &RecordBatch, index: usize) -> &Decimal128Array {
    batch.column(index).as_any().downcast_ref::<Decimal128Array>().expect("expected Decimal128Array")
}

fn date32_array(batch: &RecordBatch, index: usize) -> &Date32Array {
    batch.column(index).as_any().downcast_ref::<Date32Array>().expect("expected Date32Array")
}

fn date32(year: i32, month: u32, day: u32) -> i32 {
    let adjusted_year = year - i32::from(month <= 2);
    let era = if adjusted_year >= 0 { adjusted_year } else { adjusted_year - 399 } / 400;
    let year_of_era = adjusted_year - era * 400;
    let month = month as i32;
    let day_of_year = (153 * (month + if month > 2 { -3 } else { 9 }) + 2) / 5 + day as i32 - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * 146097 + day_of_era - 719468
}

fn timestamp_ns_array(batch: &RecordBatch, index: usize) -> &TimestampNanosecondArray {
    batch.column(index).as_any().downcast_ref::<TimestampNanosecondArray>().expect("expected TimestampNanosecondArray")
}

fn string_array(batch: &RecordBatch, index: usize) -> &StringArray {
    batch.column(index).as_any().downcast_ref::<StringArray>().expect("expected StringArray")
}

fn unique_temp_dir(prefix: &str) -> std::path::PathBuf {
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let path = std::env::temp_dir().join(format!("{prefix}-{}-{nonce}", std::process::id()));
    fs::create_dir_all(&path).unwrap();
    path
}
