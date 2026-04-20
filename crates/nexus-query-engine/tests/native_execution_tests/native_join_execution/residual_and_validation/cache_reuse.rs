use super::*;

#[test]
fn native_query_reuses_build_side_hash_join_for_repeated_broadcast_probe() {
    let dir = unique_temp_dir("nexus-query-engine-native-hash-join-cache");
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
                nexus_query_engine::expr::ColumnRef::input(0, "order_id", "left_order_id"),
                nexus_query_engine::expr::ColumnRef::input(1, "order_id", "right_order_id"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(nexus_query_engine::expr::ColumnRef::input(0, "order_id", "order_id"), "order_id"),
            JoinOutputColumn::left(nexus_query_engine::expr::ColumnRef::input(0, "qty", "qty"), "qty"),
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
                nexus_query_engine::expr::ColumnRef::input(0, "order_id", "left_order_id"),
                nexus_query_engine::expr::ColumnRef::input(1, "order_id", "right_order_id"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(nexus_query_engine::expr::ColumnRef::input(0, "qty", "qty"), "qty"),
            JoinOutputColumn::right(nexus_query_engine::expr::ColumnRef::input(1, "category", "category"), "category"),
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
    let dir = unique_temp_dir("nexus-query-engine-native-durable-hash-join-cache");
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
                JoinOutputColumn::left(nexus_query_engine::expr::ColumnRef::input(0, "qty", "qty"), "qty"),
                JoinOutputColumn::right(
                    nexus_query_engine::expr::ColumnRef::input(1, "category", "category"),
                    "category",
                ),
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
