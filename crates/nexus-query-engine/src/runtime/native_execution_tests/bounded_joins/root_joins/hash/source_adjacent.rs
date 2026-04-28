#[test]
fn bounded_source_cache_preserves_chunk_keys_for_multi_chunk_join_leaves() {
    let dir = unique_temp_dir("nexus-query-engine-bounded-source-cache-multi-chunk-join-leaves");
    let left_path = dir.join("lineitem.parquet");
    let right_path = dir.join("orders.parquet");
    let left_schema = write_row_grouped_join_left_parquet(&left_path);
    let right_schema = write_row_grouped_join_right_parquet(&right_path);
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
        Field::new("category", DataType::Utf8, false),
    ]));
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let cache = Arc::new(GpuLocalParquetSourceCache::new(64 * 1024 * 1024));

    let build_plan = || {
        let mut builder = QueryPlanBuilder::new();
        let left = builder.add_source(SourceNode::LocalParquet(join_left_source(
            &left_path,
            &left_schema,
            &["order_id", "qty"],
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
                .with_output_projection(vec![
                    JoinOutputColumn::left(
                        nexus_query_engine::expr::ColumnRef::input(0, "order_id", "order_id"),
                        "order_id",
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
        builder.finish_with_host_sink(join)
    };
    let build_context = || {
        ExecutionContext::builder()
            .with_shared_runtime(Arc::clone(&runtime))
            .with_memory_control_policy(MemoryControlPolicy::Bounded {
                device_budget_bytes: 8 * 1024 * 1024 * 1024,
                max_source_chunk_bytes: 1024 * 1024,
                max_row_groups_per_chunk: 1,
                max_retry_attempts: 1,
                min_retry_chunk_row_groups: 1,
            })
            .with_source_scheduler_policy(SourceSchedulerPolicy::new(3))
            .with_local_parquet_source_cache(Arc::clone(&cache))
            .build()
            .unwrap()
    };

    let cold = execute_query_plan(&build_context(), &build_plan()).unwrap();
    assert_eq!(cold.metrics().source_chunks_committed, 7);
    assert_eq!(cold.metrics().source_cache_hits, 0);
    assert_eq!(cold.metrics().source_cache_misses, 7);
    assert_eq!(cold.metrics().source_cache_inserts, 7);

    let warm = execute_query_plan(&build_context(), &build_plan()).unwrap();
    let metrics = warm.metrics();
    assert_eq!(metrics.source_chunks_committed, 7);
    assert_eq!(metrics.source_cache_hits, 7);
    assert_eq!(metrics.source_cache_misses, 0);
    assert_eq!(metrics.source_cache_inserts, 0);

    let QueryOutput::HostArrow(batch) = warm.output() else {
        panic!("host sink should materialize the warm multi-chunk join output");
    };
    assert_eq!(batch.num_rows(), 3);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[2, 2, 3]);
    assert_eq!(int32_array(batch, 1).values().as_ref(), &[20, 25, 30]);
    assert_eq!(string_array(batch, 2).iter().collect::<Vec<_>>(), vec![Some("two"), Some("two"), Some("three")]);

    fs::remove_dir_all(dir).unwrap();
}
