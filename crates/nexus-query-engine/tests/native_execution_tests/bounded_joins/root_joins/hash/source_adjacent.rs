#[test]
fn bounded_native_query_executes_root_hash_join_with_chunked_tokenized_leaf_reads() {
    let dir = unique_temp_dir("nexus-query-engine-bounded-root-hash-join-chunked-source");
    let left_path = dir.join("lineitem.parquet");
    let right_path = dir.join("orders.parquet");
    let left_schema = write_row_grouped_join_left_parquet(&left_path);
    let right_schema = write_row_grouped_join_right_parquet(&right_path);
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
                JoinOutputColumn::left(nexus_query_engine::expr::ColumnRef::input(0, "order_id", "order_id"), "order_id"),
                JoinOutputColumn::left(nexus_query_engine::expr::ColumnRef::input(0, "qty", "qty"), "qty"),
                JoinOutputColumn::right(nexus_query_engine::expr::ColumnRef::input(1, "category", "category"), "category"),
            ])
            .with_build_side(JoinSide::Right),
        Arc::clone(&output_schema),
    );
    let plan = builder.finish_with_host_sink(join);
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 8 * 1024 * 1024 * 1024,
            max_source_chunk_bytes: 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .with_source_scheduler_policy(SourceSchedulerPolicy::new(3))
        .build()
        .unwrap();

    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.source_chunks_planned, 7);
    assert_eq!(metrics.source_chunks_attempted, 7);
    assert_eq!(metrics.source_chunks_committed, 7);
    assert_eq!(metrics.source_read_prefetches_started, 0);
    assert_eq!(metrics.source_read_prefetches_completed, 0);
    assert_eq!(metrics.source_read_prefetches_skipped, 0);
    assert_eq!(metrics.source_read_prefetches_abandoned, 0);
    assert_eq!(metrics.reservation_denied_count, 0);
    let left_node_id = left.to_string();
    let left_source_read_reserves = metrics
        .reservation_events
        .iter()
        .filter(|event| {
            event.outcome.as_str() == "granted"
                && event.action == MemoryReservationEventAction::Reserve
                && event.reservation_kind == ReservationKind::SourceRead
                && matches!(event.memory_boundary.as_str(), "join_source_read" | "join_source_read_prefetch")
                && event.node_id.as_deref() == Some(left_node_id.as_str())
        })
        .count();
    assert_eq!(
        left_source_read_reserves, 1,
        "multi-chunk join probe source should batch adjacent row groups without an extra full-source concat reservation"
    );
    let left_source_read_bytes = metrics
        .reservation_events
        .iter()
        .filter(|event| {
            event.outcome.as_str() == "granted"
                && event.action == MemoryReservationEventAction::Reserve
                && event.reservation_kind == ReservationKind::SourceRead
                && matches!(event.memory_boundary.as_str(), "join_source_read" | "join_source_read_prefetch")
                && event.node_id.as_deref() == Some(left_node_id.as_str())
        })
        .map(|event| event.requested_bytes)
        .sum::<u64>();
    assert!(left_source_read_bytes > 0);
    assert!(
        metrics.reservation_events.iter().all(|event| {
            event.memory_boundary != "join_source_concat" || event.requested_bytes < left_source_read_bytes
        }),
        "streamed join probe leaves must not reserve all source chunks again as a full concat output"
    );
    let join_source_reads = metrics
        .reservation_events
        .iter()
        .filter(|event| {
            event.outcome.as_str() == "granted"
                && event.reservation_kind == ReservationKind::SourceRead
                && matches!(event.memory_boundary.as_str(), "join_source_read" | "join_source_read_prefetch")
        })
        .count();
    assert_eq!(join_source_reads, 2);
    let join_source_prefetches = metrics
        .reservation_events
        .iter()
        .filter(|event| {
            event.outcome.as_str() == "granted"
                && event.reservation_kind == ReservationKind::SourceRead
                && event.memory_boundary == "join_source_read_prefetch"
        })
        .count();
    assert_eq!(join_source_prefetches, 0);

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
                    JoinOutputColumn::left(nexus_query_engine::expr::ColumnRef::input(0, "order_id", "order_id"), "order_id"),
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

#[test]
fn bounded_native_query_reconciles_streamed_join_probe_prefetch_after_decode() {
    let dir = unique_temp_dir("nexus-query-engine-bounded-root-hash-join-prefetch-reconcile");
    let left_path = dir.join("lineitem.parquet");
    let right_path = dir.join("orders.parquet");
    let left_schema = write_row_grouped_join_left_with_payload_parquet(&left_path);
    let right_schema = write_row_grouped_join_right_parquet(&right_path);
    let left_row_groups = 4;
    let left_rows_per_row_group = 256;
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
        Field::new("category", DataType::Utf8, false),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let left_row_group_bytes = vec![1; left_row_groups];
    let left = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new(&left_path)
                .with_file_size_bytes(fs::metadata(&left_path).unwrap().len())
                .with_row_group_bytes(left_row_group_bytes.clone())
                .with_projected_row_group_bytes(left_row_group_bytes.clone(), left_row_group_bytes)
                .with_row_group_rows(vec![left_rows_per_row_group as u64; left_row_groups]),
        ],
        Arc::clone(&left_schema),
        vec!["order_id".to_string(), "qty".to_string(), "payload".to_string()],
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
                JoinOutputColumn::left(nexus_query_engine::expr::ColumnRef::input(0, "order_id", "order_id"), "order_id"),
                JoinOutputColumn::left(nexus_query_engine::expr::ColumnRef::input(0, "qty", "qty"), "qty"),
                JoinOutputColumn::right(nexus_query_engine::expr::ColumnRef::input(1, "category", "category"), "category"),
            ])
            .with_build_side(JoinSide::Right),
        Arc::clone(&output_schema),
    );
    let plan = builder.finish_with_host_sink(join);
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 8 * 1024 * 1024 * 1024,
            max_source_chunk_bytes: 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .with_source_scheduler_policy(SourceSchedulerPolicy::new(2))
        .build()
        .unwrap();

    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.source_chunks_planned, 7);
    assert_eq!(metrics.source_chunks_committed, 7);
    assert_eq!(metrics.source_read_prefetches_started, metrics.source_read_prefetches_completed);
    assert_eq!(metrics.source_read_prefetches_completed, 0);
    assert_eq!(metrics.source_read_prefetches_abandoned, 0);
    assert_eq!(metrics.reservation_denied_count, 0);
    assert_eq!(metrics.retry_resource_exhausted_count, 0);
    assert!(metrics.reservation_events.iter().all(|event| event.memory_boundary != "join_source_read_prefetch"));

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), left_rows_per_row_group * 3);
    assert_eq!(batch.num_columns(), 3);
    assert!(int32_array(batch, 0).values().iter().all(|order_id| matches!(*order_id, 2..=4)));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_skips_streamed_join_probe_prefetch_without_minimum_decode_headroom() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let dir = unique_temp_dir("nexus-query-engine-bounded-root-hash-join-prefetch-no-headroom");
    let left_path = dir.join("lineitem.parquet");
    let right_path = dir.join("orders.parquet");
    let left_schema = write_row_grouped_join_left_parquet(&left_path);
    let right_schema = write_row_grouped_join_right_parquet(&right_path);
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
                JoinOutputColumn::left(nexus_query_engine::expr::ColumnRef::input(0, "order_id", "order_id"), "order_id"),
                JoinOutputColumn::left(nexus_query_engine::expr::ColumnRef::input(0, "qty", "qty"), "qty"),
                JoinOutputColumn::right(nexus_query_engine::expr::ColumnRef::input(1, "category", "category"), "category"),
            ])
            .with_build_side(JoinSide::Right),
        Arc::clone(&output_schema),
    );
    let plan = builder.finish_with_host_sink(join);
    let minimum_decode_budget = u64::try_from(cudf::DataFrame::minimum_parquet_source_read_budget_bytes()).unwrap();
    assert!(minimum_decode_budget > 1);
    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 8 * 1024 * 1024 * 1024,
            max_source_chunk_bytes: minimum_decode_budget - 1,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .with_source_scheduler_policy(SourceSchedulerPolicy::new(2))
        .build()
        .unwrap();

    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.source_chunks_planned, 7);
    assert_eq!(metrics.source_chunks_committed, 7);
    assert_eq!(metrics.source_read_prefetches_started, 0);
    assert_eq!(metrics.source_read_prefetches_completed, 0);
    assert_eq!(metrics.source_read_prefetches_skipped, 0);
    assert_eq!(metrics.source_read_prefetches_abandoned, 0);
    assert_eq!(metrics.reservation_denied_count, 0);
    assert!(
        metrics.reservation_events.iter().all(|event| event.memory_boundary != "join_source_read_prefetch"),
        "optional join source prefetch should not reserve when its minimum decode reservation cannot be admitted"
    );

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 3);
    assert_eq!(batch.num_columns(), 3);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_streams_source_adjacent_filter_projection_into_hash_probe() {
    let dir = unique_temp_dir("nexus-query-engine-bounded-root-hash-join-streamed-source-unary");
    let left_path = dir.join("lineitem.parquet");
    let right_path = dir.join("orders.parquet");
    let left_schema = write_row_grouped_join_left_parquet(&left_path);
    let right_schema = write_row_grouped_join_right_parquet(&right_path);
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("category", DataType::Utf8, false),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(
        join_left_source(&left_path, &left_schema, &["order_id", "qty"])
            .with_column_roles(vec!["order_id".to_string()], vec!["qty".to_string()]),
    ));
    let filtered_left = builder.add_filter(
        left,
        Expr::binary(Expr::column("qty"), BinaryOp::Gt, Expr::literal(LiteralValue::Int32(15))),
        Arc::clone(&left_schema),
    );
    let projected_left = builder.add_projection(
        filtered_left,
        vec![ProjectionOutput::new(Expr::column("order_id"), Field::new("order_id", DataType::Int32, false))],
    );
    let right = builder.add_source(SourceNode::LocalParquet(join_right_source(
        &right_path,
        &right_schema,
        &["order_id", "category"],
    )));
    let join = builder.add_join(
        projected_left,
        right,
        JoinSpec::new(JoinType::Inner, vec![join_order_id_key()])
            .with_output_projection(vec![
                JoinOutputColumn::left(nexus_query_engine::expr::ColumnRef::input(0, "order_id", "order_id"), "order_id"),
                JoinOutputColumn::right(nexus_query_engine::expr::ColumnRef::input(1, "category", "category"), "category"),
            ])
            .with_build_side(JoinSide::Right),
        Arc::clone(&output_schema),
    );
    let plan = builder.finish_with_host_sink(join);
    assert!(
        plan.capability_analysis().admits_strict_bounded_join(),
        "source-adjacent filter/projection hash probe plan should stay capability-admitted:\n{plan}"
    );
    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::new(cudf::Runtime::new().unwrap()))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 8 * 1024 * 1024 * 1024,
            max_source_chunk_bytes: 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .with_source_scheduler_policy(SourceSchedulerPolicy::new(3))
        .build()
        .unwrap();

    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.source_chunks_planned, 7);
    assert_eq!(metrics.source_chunks_committed, 7);
    assert_eq!(metrics.reservation_denied_count, 0);
    let left_node_id = left.to_string();
    let left_source_read_reserves = metrics
        .reservation_events
        .iter()
        .filter(|event| {
            event.outcome.as_str() == "granted"
                && event.action == MemoryReservationEventAction::Reserve
                && event.reservation_kind == ReservationKind::SourceRead
                && matches!(event.memory_boundary.as_str(), "join_source_read" | "join_source_read_prefetch")
                && event.node_id.as_deref() == Some(left_node_id.as_str())
        })
        .count();
    assert_eq!(left_source_read_reserves, 1);
    assert!(
        metrics.reservation_events.iter().all(|event| event.memory_boundary != "join_source_concat"),
        "source-adjacent unary probe stream should not materialize and concat the full source before probing"
    );
    let streamed_filter_outputs = metrics
        .reservation_events
        .iter()
        .filter(|event| {
            event.outcome.as_str() == "applied"
                && event.action == MemoryReservationEventAction::Transfer
                && event.reservation_kind == ReservationKind::OperatorOutput
                && event.memory_boundary == "join_source_filter_output"
        })
        .count();
    let streamed_projection_outputs = metrics
        .reservation_events
        .iter()
        .filter(|event| {
            event.outcome.as_str() == "applied"
                && event.action == MemoryReservationEventAction::Transfer
                && event.reservation_kind == ReservationKind::OperatorOutput
                && event.memory_boundary == "join_source_projection_output"
        })
        .count();
    assert_eq!(streamed_filter_outputs, 1);
    assert_eq!(streamed_projection_outputs, 1);
    assert!(
        metrics.reservation_events.iter().all(|event| {
            event.memory_boundary != "compiled_filter_output" && event.memory_boundary != "compiled_projection_output"
        }),
        "source-adjacent filter/projection should be fused into the source stream consumer path"
    );

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 3);
    assert_eq!(batch.num_columns(), 2);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[2, 2, 3]);
    assert_eq!(string_array(batch, 1).iter().collect::<Vec<_>>(), vec![Some("two"), Some("two"), Some("three")]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_executes_source_adjacent_join_filter_predicate() {
    let dir = unique_temp_dir("nexus-query-engine-bounded-root-hash-join-source-filter");
    let left_path = dir.join("lineitem_part.parquet");
    let right_path = dir.join("part_category.parquet");
    let left_schema = write_source_adjacent_join_filter_left_parquet(&left_path);
    let right_schema = write_source_adjacent_join_filter_right_parquet(&right_path);
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("partkey", DataType::Int32, false),
        Field::new("category", DataType::Utf8, false),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(
        join_left_source(
            &left_path,
            &left_schema,
            &["partkey", "brand", "container", "quantity", "size", "shipmode", "shipinstruct"],
        )
        .with_column_roles(
            vec!["partkey".to_string()],
            vec![
                "brand".to_string(),
                "container".to_string(),
                "quantity".to_string(),
                "size".to_string(),
                "shipmode".to_string(),
                "shipinstruct".to_string(),
            ],
        ),
    ));
    let filtered_left = builder.add_filter(left, source_adjacent_join_filter_predicate(), Arc::clone(&left_schema));
    let projected_left = builder.add_projection(
        filtered_left,
        vec![ProjectionOutput::new(Expr::column("partkey"), Field::new("partkey", DataType::Int32, false))],
    );
    let right = builder.add_source(SourceNode::LocalParquet(join_right_source(
        &right_path,
        &right_schema,
        &["partkey", "category"],
    )));
    let join = builder.add_join(
        projected_left,
        right,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                nexus_query_engine::expr::ColumnRef::input(0, "partkey", "left_partkey"),
                nexus_query_engine::expr::ColumnRef::input(1, "partkey", "right_partkey"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(nexus_query_engine::expr::ColumnRef::input(0, "partkey", "partkey"), "partkey"),
            JoinOutputColumn::right(nexus_query_engine::expr::ColumnRef::input(1, "category", "category"), "category"),
        ])
        .with_build_side(JoinSide::Right),
        Arc::clone(&output_schema),
    );
    let plan = builder.finish_with_host_sink(join);
    assert!(
        plan.capability_analysis().admits_strict_bounded_join(),
        "source-adjacent filter should stay capability-admitted:\n{plan}"
    );
    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::new(cudf::Runtime::new().unwrap()))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 8 * 1024 * 1024 * 1024,
            max_source_chunk_bytes: 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .with_source_scheduler_policy(SourceSchedulerPolicy::new(2))
        .build()
        .unwrap();

    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.reservation_denied_count, 0);
    assert!(
        metrics.reservation_events.iter().any(|event| {
            event.outcome.as_str() == "applied"
                && event.action == MemoryReservationEventAction::Transfer
                && event.reservation_kind == ReservationKind::OperatorOutput
                && event.memory_boundary == "join_source_filter_output"
        }),
        "source-adjacent filter must execute on the join_source_filter reservation boundary"
    );

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 3);
    assert_eq!(batch.num_columns(), 2);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[1, 2, 3]);
    assert_eq!(
        string_array(batch, 1).iter().collect::<Vec<_>>(),
        vec![Some("brand12"), Some("brand23"), Some("brand34")]
    );

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_defers_single_chunk_source_adjacent_filter_projection_into_hash_probe() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let dir = unique_temp_dir("nexus-query-engine-bounded-root-hash-join-single-source-unary");
    let left_path = dir.join("lineitem.parquet");
    let right_path = dir.join("orders.parquet");
    let left_schema = write_row_grouped_join_left_parquet(&left_path);
    let right_schema = write_join_right_parquet(&runtime, &right_path);
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("category", DataType::Utf8, false),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(
        join_left_source(&left_path, &left_schema, &["order_id", "qty"])
            .with_column_roles(vec!["order_id".to_string()], vec!["qty".to_string()]),
    ));
    let filtered_left = builder.add_filter(
        left,
        Expr::binary(Expr::column("qty"), BinaryOp::Gt, Expr::literal(LiteralValue::Int32(15))),
        Arc::clone(&left_schema),
    );
    let projected_left = builder.add_projection(
        filtered_left,
        vec![ProjectionOutput::new(Expr::column("order_id"), Field::new("order_id", DataType::Int32, false))],
    );
    let right = builder.add_source(SourceNode::LocalParquet(join_right_source(
        &right_path,
        &right_schema,
        &["order_id", "category"],
    )));
    let join = builder.add_join(
        projected_left,
        right,
        JoinSpec::new(JoinType::Inner, vec![join_order_id_key()])
            .with_output_projection(vec![
                JoinOutputColumn::left(nexus_query_engine::expr::ColumnRef::input(0, "order_id", "order_id"), "order_id"),
                JoinOutputColumn::right(nexus_query_engine::expr::ColumnRef::input(1, "category", "category"), "category"),
            ])
            .with_build_side(JoinSide::Right),
        Arc::clone(&output_schema),
    );
    let plan = builder.finish_with_host_sink(join);
    assert!(
        plan.capability_analysis().admits_strict_bounded_join(),
        "single-chunk source-adjacent filter/projection hash probe plan should stay capability-admitted:\n{plan}"
    );
    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 8 * 1024 * 1024 * 1024,
            max_source_chunk_bytes: 1024 * 1024 * 1024,
            max_row_groups_per_chunk: 1024,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .with_source_scheduler_policy(SourceSchedulerPolicy::new(3))
        .build()
        .unwrap();

    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.source_chunks_planned, 2);
    assert_eq!(metrics.source_chunks_committed, 2);
    assert_eq!(metrics.reservation_denied_count, 0);
    let left_node_id = left.to_string();
    assert!(
        metrics.reservation_events.iter().all(|event| {
            !(event.memory_boundary == "join_source_read"
                && event.node_id.as_deref() == Some(left_node_id.as_str())
                && event.action == MemoryReservationEventAction::Grow)
        }),
        "single-chunk source-adjacent join input should not grow the raw source-read reservation before filtering"
    );
    assert_eq!(
        metrics
            .reservation_events
            .iter()
            .filter(|event| {
                event.outcome.as_str() == "applied"
                    && event.action == MemoryReservationEventAction::Transfer
                    && event.reservation_kind == ReservationKind::OperatorOutput
                    && event.memory_boundary == "join_source_filter_output"
            })
            .count(),
        1
    );
    assert_eq!(
        metrics
            .reservation_events
            .iter()
            .filter(|event| {
                event.outcome.as_str() == "applied"
                    && event.action == MemoryReservationEventAction::Transfer
                    && event.reservation_kind == ReservationKind::OperatorOutput
                    && event.memory_boundary == "join_source_projection_output"
            })
            .count(),
        1
    );
    assert!(
        metrics.reservation_events.iter().all(|event| {
            event.memory_boundary != "compiled_filter_output" && event.memory_boundary != "compiled_projection_output"
        }),
        "single-chunk source-adjacent filter/projection should be fused into the source stream consumer path"
    );

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 3);
    assert_eq!(batch.num_columns(), 2);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[2, 2, 3]);
    assert_eq!(string_array(batch, 1).iter().collect::<Vec<_>>(), vec![Some("two"), Some("two"), Some("three")]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_streams_source_adjacent_filter_projection_into_hash_build() {
    let dir = unique_temp_dir("nexus-query-engine-bounded-root-hash-join-streamed-build-source-unary");
    let left_path = dir.join("lineitem.parquet");
    let right_path = dir.join("orders.parquet");
    let left_schema = write_row_grouped_join_left_parquet(&left_path);
    let right_schema = write_row_grouped_join_right_parquet(&right_path);
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("category", DataType::Utf8, false),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(
        join_left_source(&left_path, &left_schema, &["order_id", "qty"])
            .with_column_roles(vec!["order_id".to_string()], vec!["qty".to_string()]),
    ));
    let filtered_left = builder.add_filter(
        left,
        Expr::binary(Expr::column("qty"), BinaryOp::Gt, Expr::literal(LiteralValue::Int32(15))),
        Arc::clone(&left_schema),
    );
    let projected_left = builder.add_projection(
        filtered_left,
        vec![ProjectionOutput::new(Expr::column("order_id"), Field::new("order_id", DataType::Int32, false))],
    );
    let right = builder.add_source(SourceNode::LocalParquet(join_right_source(
        &right_path,
        &right_schema,
        &["order_id", "category"],
    )));
    let join = builder.add_join(
        projected_left,
        right,
        JoinSpec::new(JoinType::Inner, vec![join_order_id_key()])
            .with_output_projection(vec![
                JoinOutputColumn::left(nexus_query_engine::expr::ColumnRef::input(0, "order_id", "order_id"), "order_id"),
                JoinOutputColumn::right(nexus_query_engine::expr::ColumnRef::input(1, "category", "category"), "category"),
            ])
            .with_build_side(JoinSide::Left),
        Arc::clone(&output_schema),
    );
    let plan = builder.finish_with_host_sink(join);
    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::new(cudf::Runtime::new().unwrap()))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 8 * 1024 * 1024 * 1024,
            max_source_chunk_bytes: 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .with_source_scheduler_policy(SourceSchedulerPolicy::new(3))
        .build()
        .unwrap();

    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.source_chunks_planned, 7);
    assert_eq!(metrics.source_chunks_committed, 7);
    assert_eq!(metrics.reservation_denied_count, 0);
    assert!(
        metrics.reservation_events.iter().all(|event| event.memory_boundary != "join_source_concat"),
        "source-adjacent unary build stream should not concat the raw source before filtering/projection"
    );
    assert_eq!(
        metrics
            .reservation_events
            .iter()
            .filter(|event| {
                event.outcome.as_str() == "applied"
                    && event.action == MemoryReservationEventAction::Transfer
                    && event.reservation_kind == ReservationKind::OperatorOutput
                    && event.memory_boundary == "join_source_filter_output"
            })
            .count(),
        1
    );
    assert_eq!(
        metrics
            .reservation_events
            .iter()
            .filter(|event| {
                event.outcome.as_str() == "applied"
                    && event.action == MemoryReservationEventAction::Transfer
                    && event.reservation_kind == ReservationKind::OperatorOutput
                    && event.memory_boundary == "join_source_projection_output"
            })
            .count(),
        1
    );

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 3);
    assert_eq!(batch.num_columns(), 2);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[2, 2, 3]);
    assert_eq!(string_array(batch, 1).iter().collect::<Vec<_>>(), vec![Some("two"), Some("two"), Some("three")]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_streams_source_adjacent_filter_projection_into_partitioned_probe_materialization() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let dir = unique_temp_dir("nexus-query-engine-bounded-one-shot-probe-streamed-source-unary");
    let left_path = dir.join("lineitem.parquet");
    let right_path = dir.join("orders.parquet");
    let left_schema = write_row_grouped_join_left_parquet(&left_path);
    let right_schema = write_join_right_parquet(&runtime, &right_path);
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("category", DataType::Utf8, false),
    ]));

    let mut right_source = join_right_source(&right_path, &right_schema, &["order_id", "category"]);
    right_source.files[0].row_group_rows = Some(vec![100_000_000]);

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(
        join_left_source(&left_path, &left_schema, &["order_id", "qty"])
            .with_column_roles(vec!["order_id".to_string()], vec!["qty".to_string()]),
    ));
    let filtered_left = builder.add_filter(
        left,
        Expr::binary(Expr::column("qty"), BinaryOp::Gt, Expr::literal(LiteralValue::Int32(15))),
        Arc::clone(&left_schema),
    );
    let projected_left = builder.add_projection(
        filtered_left,
        vec![ProjectionOutput::new(Expr::column("order_id"), Field::new("order_id", DataType::Int32, false))],
    );
    let right = builder.add_source(SourceNode::LocalParquet(right_source));
    let join = builder.add_join(
        projected_left,
        right,
        JoinSpec::new(JoinType::Inner, vec![join_order_id_key()])
            .with_output_projection(vec![
                JoinOutputColumn::left(nexus_query_engine::expr::ColumnRef::input(0, "order_id", "order_id"), "order_id"),
                JoinOutputColumn::right(nexus_query_engine::expr::ColumnRef::input(1, "category", "category"), "category"),
            ])
            .with_build_side(JoinSide::Right),
        Arc::clone(&output_schema),
    );
    let plan = builder.finish_with_host_sink(join);
    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 8 * 1024 * 1024 * 1024,
            max_source_chunk_bytes: 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .with_source_scheduler_policy(SourceSchedulerPolicy::new(3))
        .build()
        .unwrap();

    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.one_shot_join_count, 0);
    let join_metrics =
        metrics.operators.iter().find_map(|operator| operator.join.as_ref()).expect("join metrics should be recorded");
    assert_eq!(join_metrics.execution_strategy, "bounded_partitioned_hash_equi_join");
    assert_eq!(metrics.reservation_denied_count, 0);
    assert_eq!(metrics.source_read_prefetches_started, 0);
    assert_eq!(metrics.source_read_prefetches_completed, 0);
    assert_eq!(metrics.source_read_prefetches_skipped, 0);
    assert_eq!(metrics.source_read_prefetches_abandoned, 0);
    assert!(
        metrics.reservation_events.iter().all(|event| event.memory_boundary != "join_source_read_prefetch"),
        "partitioned source-adjacent materialization should not prefetch while transformed chunks are retained"
    );
    assert!(
        metrics.reservation_events.iter().all(|event| event.memory_boundary != "join_source_concat_output"),
        "partitioned source-adjacent probe streaming should not concatenate the source before join materialization"
    );
    assert!(
        metrics.reservation_events.iter().any(|event| {
            event.memory_boundary == "join_output_stream_concat"
                && event.action == MemoryReservationEventAction::Reserve
                && event.reservation_kind == ReservationKind::JoinOutput
        }),
        "partitioned source-adjacent probe streaming should concatenate join output chunks at the output boundary"
    );
    assert!(
        metrics.reservation_events.iter().any(|event| {
            event.memory_boundary == "join_hash_partitioned_left_side"
                && event.action == MemoryReservationEventAction::Reserve
                && event.reservation_kind == ReservationKind::OperatorOutput
        }),
        "partitioned probe streaming should hash-partition each streamed probe chunk"
    );
    assert_eq!(
        metrics
            .reservation_events
            .iter()
            .filter(|event| {
                event.outcome.as_str() == "applied"
                    && event.action == MemoryReservationEventAction::Transfer
                    && event.reservation_kind == ReservationKind::OperatorOutput
                    && event.memory_boundary == "join_source_filter_output"
            })
            .count(),
        1
    );
    assert_eq!(
        metrics
            .reservation_events
            .iter()
            .filter(|event| {
                event.outcome.as_str() == "applied"
                    && event.action == MemoryReservationEventAction::Transfer
                    && event.reservation_kind == ReservationKind::OperatorOutput
                    && event.memory_boundary == "join_source_projection_output"
            })
            .count(),
        1
    );
    assert!(
        metrics.reservation_events.iter().all(|event| {
            event.memory_boundary != "compiled_filter_output" && event.memory_boundary != "compiled_projection_output"
        }),
        "source-adjacent partitioned probe filtering/projection should be fused into the join source stream"
    );
    let first_projection_transfer = metrics.reservation_events.iter().position(|event| {
        event.outcome.as_str() == "applied"
            && event.action == MemoryReservationEventAction::Transfer
            && event.reservation_kind == ReservationKind::OperatorOutput
            && event.memory_boundary == "join_source_projection_output"
    });
    assert!(
        first_projection_transfer.is_some(),
        "partitioned probe stream should still filter/project the batched source stream"
    );

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 3);
    assert_eq!(batch.num_columns(), 2);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[2, 2, 3]);
    assert_eq!(string_array(batch, 1).iter().collect::<Vec<_>>(), vec![Some("two"), Some("two"), Some("three")]);

    fs::remove_dir_all(dir).unwrap();
}
