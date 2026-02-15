use super::*;

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
    assert!(metrics.reservation_live_peak_bytes > 0);
    assert_eq!(metrics.hash_join_build_count, 1);
    assert!(metrics.hash_join_hash_table_bytes > 0);
    let join_metrics =
        metrics.operators.iter().find_map(|operator| operator.join.as_ref()).expect("join metrics should be recorded");
    assert_eq!(join_metrics.join_type, "inner");
    assert_eq!(join_metrics.build_side, "right");
    assert_eq!(join_metrics.probe_side, "left");
    assert!(join_metrics.gather_map_bytes > 0);
    assert_eq!(join_metrics.output_rows, 3);
    assert_granted_reservation_trace(
        metrics,
        &[
            (ReservationKind::SourceRead, "join_source_read"),
            (ReservationKind::SourceRead, "join_source_read"),
            (ReservationKind::BuildSide, "join_build_side"),
            (ReservationKind::HashTable, "join_hash_table"),
            (ReservationKind::ProbeSide, "join_probe_side"),
            (ReservationKind::GatherMap, "join_gather_map"),
            (ReservationKind::JoinOutput, "join_output"),
            (ReservationKind::Materialization, "host_materialization"),
        ],
    );
    assert!(
        metrics.reservation_events.iter().all(|event| event.memory_boundary != "join_input_projection"),
        "schema-only join input selection must alias or transfer the resident input instead of charging a projection"
    );

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
fn bounded_native_query_routes_large_keyed_inner_build_to_partitioned_hash_join() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let dir = unique_temp_dir("cudf-query-engine-bounded-large-keyed-inner-partitioned-hash");
    let left_path = dir.join("lineitem.parquet");
    let right_path = dir.join("orders.parquet");
    let left_schema = write_join_left_parquet(&runtime, &left_path);
    let right_schema = write_join_right_parquet(&runtime, &right_path);
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
        Field::new("category", DataType::Utf8, false),
    ]));

    let mut right_source = join_right_source(&right_path, &right_schema, &["order_id", "category"]);
    right_source.files[0].row_group_rows = Some(vec![100_000_000]);

    let mut builder = QueryPlanBuilder::new();
    let left =
        builder.add_source(SourceNode::LocalParquet(join_left_source(&left_path, &left_schema, &["order_id", "qty"])));
    let right = builder.add_source(SourceNode::LocalParquet(right_source));
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
    assert_eq!(metrics.hash_join_build_count, 0);
    assert_eq!(metrics.one_shot_join_count, 0);
    assert!(
        metrics.reservation_events.iter().all(|event| event.reservation_kind != ReservationKind::HashTable),
        "large partitioned join should not reserve reusable hash-table memory"
    );
    let join_metrics =
        metrics.operators.iter().find_map(|operator| operator.join.as_ref()).expect("join metrics should be recorded");
    assert_eq!(join_metrics.execution_strategy, "bounded_partitioned_hash_equi_join");
    assert!(join_metrics.gather_map_bytes > 0);
    assert_eq!(join_metrics.output_rows, 3);
    assert!(metrics.join_strategy_decisions.iter().any(|decision| {
        decision.node_id == "n2"
            && decision.facts.get("join_strategy").map(String::as_str) == Some("partitioned_hash_equi")
            && decision.facts.get("partition_count").map(String::as_str) == Some("2")
            && decision.facts.get("partition_target_bytes").map(String::as_str) == Some("134217728")
    }));

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
fn bounded_native_query_routes_large_keyed_inner_residual_build_to_partitioned_mixed_join() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let dir = unique_temp_dir("cudf-query-engine-bounded-large-keyed-inner-residual-one-shot");
    let left_path = dir.join("lineitem.parquet");
    let right_path = dir.join("orders.parquet");
    let left_schema = write_join_left_parquet(&runtime, &left_path);
    let right_schema = write_join_right_parquet(&runtime, &right_path);
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
        Field::new("category", DataType::Utf8, false),
    ]));

    let mut right_source = join_right_source(&right_path, &right_schema, &["order_id", "category"]);
    right_source.files[0].row_group_rows = Some(vec![100_000_000]);

    let mut builder = QueryPlanBuilder::new();
    let left =
        builder.add_source(SourceNode::LocalParquet(join_left_source(&left_path, &left_schema, &["order_id", "qty"])));
    let right = builder.add_source(SourceNode::LocalParquet(right_source));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(JoinType::Inner, vec![join_order_id_key()])
            .with_residual_predicate(qty_gt_right_order_id_times_ten())
            .with_output_projection(vec![
                JoinOutputColumn::left(cudf_query_engine::ColumnRef::input(0, "order_id", "order_id"), "order_id"),
                JoinOutputColumn::left(cudf_query_engine::ColumnRef::input(0, "qty", "qty"), "qty"),
                JoinOutputColumn::right(cudf_query_engine::ColumnRef::input(1, "category", "category"), "category"),
            ])
            .with_build_side(JoinSide::Right),
        Arc::clone(&output_schema),
    );
    let plan = builder.finish_with_host_sink(join);
    let context = bounded_join_execution_context(&runtime);

    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.hash_join_build_count, 0);
    assert_eq!(metrics.one_shot_join_count, 0);
    assert!(metrics.mixed_join_size_data_bytes > 0);
    assert!(
        metrics.reservation_events.iter().all(|event| !matches!(event.reservation_kind, ReservationKind::HashTable)),
        "large partitioned residual join should not reserve reusable hash-table memory"
    );
    assert!(metrics.reservation_events.iter().any(|event| {
        event.outcome.as_str() == "granted" && event.memory_boundary == "join_partition_mixed_size_data"
    }));
    assert!(metrics.reservation_events.iter().any(|event| {
        event.outcome.as_str() == "granted" && event.memory_boundary == "join_partition_mixed_indices"
    }));
    let join_metrics =
        metrics.operators.iter().find_map(|operator| operator.join.as_ref()).expect("join metrics should be recorded");
    assert_eq!(join_metrics.execution_strategy, "bounded_partitioned_mixed_keyed_residual_join");
    assert!(join_metrics.residual_predicate);
    assert_eq!(join_metrics.output_rows, 1);
    assert!(metrics.join_strategy_decisions.iter().any(|decision| {
        decision.node_id == "n2"
            && decision.facts.get("join_strategy").map(String::as_str) == Some("partitioned_mixed_keyed_residual")
            && decision.facts.get("partition_count").map(String::as_str) == Some("2")
    }));

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 1);
    assert_eq!(batch.num_columns(), 3);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[2]);
    assert_eq!(int32_array(batch, 1).values().as_ref(), &[25]);
    assert_eq!(string_array(batch, 2).iter().collect::<Vec<_>>(), vec![Some("two")]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_routes_large_keyed_left_build_to_partitioned_hash_with_null_padding() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let dir = unique_temp_dir("cudf-query-engine-bounded-large-keyed-left-partitioned-hash");
    let left_path = dir.join("lineitem.parquet");
    let right_path = dir.join("orders.parquet");
    let left_schema = write_join_left_parquet(&runtime, &left_path);
    let right_schema = write_join_right_keys_parquet(&runtime, &right_path, &[2, 2, 4]);
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
        Field::new("category", DataType::Utf8, true),
    ]));

    let mut right_source = join_right_source(&right_path, &right_schema, &["order_id", "category"]);
    right_source.files[0].row_group_rows = Some(vec![100_000_000]);

    let mut builder = QueryPlanBuilder::new();
    let left =
        builder.add_source(SourceNode::LocalParquet(join_left_source(&left_path, &left_schema, &["order_id", "qty"])));
    let right = builder.add_source(SourceNode::LocalParquet(right_source));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(JoinType::Left, vec![join_order_id_key()])
            .with_output_projection(vec![
                JoinOutputColumn::left(cudf_query_engine::ColumnRef::input(0, "order_id", "order_id"), "order_id"),
                JoinOutputColumn::left(cudf_query_engine::ColumnRef::input(0, "qty", "qty"), "qty"),
                JoinOutputColumn::right(cudf_query_engine::ColumnRef::input(1, "category", "category"), "category"),
            ])
            .with_build_side(JoinSide::Right),
        Arc::clone(&output_schema),
    );
    let sort = builder.add_sort(
        join,
        SortSpec::new(vec![
            SortExpr::new(Expr::column("order_id"), false, false),
            SortExpr::new(Expr::column("qty"), false, false),
            SortExpr::new(Expr::column("category"), false, false),
        ]),
        Arc::clone(&output_schema),
    );
    let plan = builder.finish_with_host_sink(sort);
    let context = bounded_join_execution_context(&runtime);

    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.hash_join_build_count, 0);
    assert_eq!(metrics.one_shot_join_count, 0);
    assert!(
        metrics.reservation_events.iter().all(|event| event.reservation_kind != ReservationKind::HashTable),
        "large partitioned left join should not reserve reusable hash-table memory"
    );
    let join_metrics =
        metrics.operators.iter().find_map(|operator| operator.join.as_ref()).expect("join metrics should be recorded");
    assert_eq!(join_metrics.join_type, "left");
    assert_eq!(join_metrics.execution_strategy, "bounded_partitioned_hash_equi_join");
    assert_eq!(join_metrics.output_rows, 6);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 6);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[1, 2, 2, 2, 2, 3]);
    assert_eq!(int32_array(batch, 1).values().as_ref(), &[10, 20, 20, 25, 25, 30]);
    let category = string_array(batch, 2);
    assert!(category.is_null(0));
    assert_eq!(category.value(1), "category_2");
    assert_eq!(category.value(2), "category_2");
    assert_eq!(category.value(3), "category_2");
    assert_eq!(category.value(4), "category_2");
    assert!(category.is_null(5));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_routes_large_keyed_full_build_to_partitioned_hash_with_null_padding() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let dir = unique_temp_dir("cudf-query-engine-bounded-large-keyed-full-partitioned-hash");
    let left_path = dir.join("lineitem.parquet");
    let right_path = dir.join("orders.parquet");
    let left_schema = write_join_left_parquet(&runtime, &left_path);
    let right_schema = write_join_right_keys_parquet(&runtime, &right_path, &[2, 2, 4]);
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("left_order_id", DataType::Int32, true),
        Field::new("right_order_id", DataType::Int32, true),
        Field::new("qty", DataType::Int32, true),
        Field::new("category", DataType::Utf8, true),
    ]));

    let mut right_source = join_right_source(&right_path, &right_schema, &["order_id", "category"]);
    right_source.files[0].row_group_rows = Some(vec![100_000_000]);

    let mut builder = QueryPlanBuilder::new();
    let left =
        builder.add_source(SourceNode::LocalParquet(join_left_source(&left_path, &left_schema, &["order_id", "qty"])));
    let right = builder.add_source(SourceNode::LocalParquet(right_source));
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
            .with_build_side(JoinSide::Right),
        Arc::clone(&output_schema),
    );
    let sort = builder.add_sort(
        join,
        SortSpec::new(vec![
            SortExpr::new(Expr::column("left_order_id"), false, false),
            SortExpr::new(Expr::column("qty"), false, false),
            SortExpr::new(Expr::column("right_order_id"), false, false),
            SortExpr::new(Expr::column("category"), false, false),
        ]),
        Arc::clone(&output_schema),
    );
    let plan = builder.finish_with_host_sink(sort);
    let context = bounded_join_execution_context(&runtime);

    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.hash_join_build_count, 0);
    assert_eq!(metrics.one_shot_join_count, 0);
    assert!(
        metrics.reservation_events.iter().all(|event| event.reservation_kind != ReservationKind::HashTable),
        "large partitioned full join should not reserve reusable hash-table memory"
    );
    let join_metrics =
        metrics.operators.iter().find_map(|operator| operator.join.as_ref()).expect("join metrics should be recorded");
    assert_eq!(join_metrics.join_type, "full");
    assert_eq!(join_metrics.execution_strategy, "bounded_partitioned_hash_equi_join");
    assert_eq!(join_metrics.output_rows, 7);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 7);
    let left_order_id = int32_array(batch, 0);
    assert_eq!(left_order_id.value(0), 1);
    assert_eq!(left_order_id.value(1), 2);
    assert_eq!(left_order_id.value(2), 2);
    assert_eq!(left_order_id.value(3), 2);
    assert_eq!(left_order_id.value(4), 2);
    assert_eq!(left_order_id.value(5), 3);
    assert!(left_order_id.is_null(6));

    let right_order_id = int32_array(batch, 1);
    assert!(right_order_id.is_null(0));
    assert_eq!(right_order_id.value(1), 2);
    assert_eq!(right_order_id.value(2), 2);
    assert_eq!(right_order_id.value(3), 2);
    assert_eq!(right_order_id.value(4), 2);
    assert!(right_order_id.is_null(5));
    assert_eq!(right_order_id.value(6), 4);

    let qty = int32_array(batch, 2);
    assert_eq!(qty.value(0), 10);
    assert_eq!(qty.value(1), 20);
    assert_eq!(qty.value(2), 20);
    assert_eq!(qty.value(3), 25);
    assert_eq!(qty.value(4), 25);
    assert_eq!(qty.value(5), 30);
    assert!(qty.is_null(6));

    let category = string_array(batch, 3);
    assert!(category.is_null(0));
    assert_eq!(category.value(1), "category_2");
    assert_eq!(category.value(2), "category_2");
    assert_eq!(category.value(3), "category_2");
    assert_eq!(category.value(4), "category_2");
    assert!(category.is_null(5));
    assert_eq!(category.value(6), "category_4");

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_executes_root_hash_join_with_chunked_tokenized_leaf_reads() {
    let dir = unique_temp_dir("cudf-query-engine-bounded-root-hash-join-chunked-source");
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
                JoinOutputColumn::left(cudf_query_engine::ColumnRef::input(0, "order_id", "order_id"), "order_id"),
                JoinOutputColumn::left(cudf_query_engine::ColumnRef::input(0, "qty", "qty"), "qty"),
                JoinOutputColumn::right(cudf_query_engine::ColumnRef::input(1, "category", "category"), "category"),
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
    assert_eq!(metrics.source_read_prefetches_started, 5);
    assert_eq!(metrics.source_read_prefetches_completed, 5);
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
        left_source_read_reserves, 4,
        "multi-chunk join probe source should stream chunk reads without an extra full-source concat reservation"
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
    assert!(join_source_reads >= 7);
    let join_source_prefetches = metrics
        .reservation_events
        .iter()
        .filter(|event| {
            event.outcome.as_str() == "granted"
                && event.reservation_kind == ReservationKind::SourceRead
                && event.memory_boundary == "join_source_read_prefetch"
        })
        .count();
    assert_eq!(join_source_prefetches, 5);

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
fn bounded_native_query_streams_source_adjacent_filter_projection_into_hash_probe() {
    let dir = unique_temp_dir("cudf-query-engine-bounded-root-hash-join-streamed-source-unary");
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
                JoinOutputColumn::left(cudf_query_engine::ColumnRef::input(0, "order_id", "order_id"), "order_id"),
                JoinOutputColumn::right(cudf_query_engine::ColumnRef::input(1, "category", "category"), "category"),
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
    assert_eq!(left_source_read_reserves, 4);
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
    assert_eq!(streamed_filter_outputs, 4);
    assert_eq!(streamed_projection_outputs, 4);
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
fn bounded_native_query_defers_single_chunk_source_adjacent_filter_projection_into_hash_probe() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let dir = unique_temp_dir("cudf-query-engine-bounded-root-hash-join-single-source-unary");
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
                JoinOutputColumn::left(cudf_query_engine::ColumnRef::input(0, "order_id", "order_id"), "order_id"),
                JoinOutputColumn::right(cudf_query_engine::ColumnRef::input(1, "category", "category"), "category"),
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
    let dir = unique_temp_dir("cudf-query-engine-bounded-root-hash-join-streamed-build-source-unary");
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
                JoinOutputColumn::left(cudf_query_engine::ColumnRef::input(0, "order_id", "order_id"), "order_id"),
                JoinOutputColumn::right(cudf_query_engine::ColumnRef::input(1, "category", "category"), "category"),
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
        4
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
        4
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
    let dir = unique_temp_dir("cudf-query-engine-bounded-one-shot-probe-streamed-source-unary");
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
                JoinOutputColumn::left(cudf_query_engine::ColumnRef::input(0, "order_id", "order_id"), "order_id"),
                JoinOutputColumn::right(cudf_query_engine::ColumnRef::input(1, "category", "category"), "category"),
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
        4
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
        4
    );
    assert!(
        metrics.reservation_events.iter().all(|event| {
            event.memory_boundary != "compiled_filter_output" && event.memory_boundary != "compiled_projection_output"
        }),
        "source-adjacent partitioned probe filtering/projection should be fused into the join source stream"
    );
    let left_node_id = left.to_string();
    let second_left_read = metrics
        .reservation_events
        .iter()
        .enumerate()
        .filter(|(_index, event)| {
            event.outcome.as_str() == "granted"
                && event.action == MemoryReservationEventAction::Reserve
                && event.reservation_kind == ReservationKind::SourceRead
                && matches!(event.memory_boundary.as_str(), "join_source_read" | "join_source_read_prefetch")
                && event.node_id.as_deref() == Some(left_node_id.as_str())
        })
        .nth(1)
        .map(|(index, _event)| index);
    let first_projection_transfer = metrics.reservation_events.iter().position(|event| {
        event.outcome.as_str() == "applied"
            && event.action == MemoryReservationEventAction::Transfer
            && event.reservation_kind == ReservationKind::OperatorOutput
            && event.memory_boundary == "join_source_projection_output"
    });
    assert!(
        first_projection_transfer.zip(second_left_read).is_some_and(|(projection, read)| projection < read),
        "partitioned probe stream should filter/project the first source chunk before reading later chunks"
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
fn bounded_native_query_preserves_partitioned_join_stream_through_filtered_join_into_aggregate() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let dir = unique_temp_dir("cudf-query-engine-bounded-one-shot-filtered-aggregate-stream");
    let left_path = dir.join("lineitem.parquet");
    let right_path = dir.join("orders.parquet");
    let semi_path = dir.join("semi_orders.parquet");
    let left_schema = write_row_grouped_join_left_parquet(&left_path);
    let right_schema = write_join_right_parquet(&runtime, &right_path);
    let semi_schema = write_join_right_keys_parquet(&runtime, &semi_path, &[2, 3]);
    let joined_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("category", DataType::Utf8, false),
    ]));
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("category", DataType::Utf8, false),
        Field::new("order_count", DataType::Int64, false),
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
    let joined = builder.add_join(
        projected_left,
        right,
        JoinSpec::new(JoinType::Inner, vec![join_order_id_key()])
            .with_output_projection(vec![
                JoinOutputColumn::left(cudf_query_engine::ColumnRef::input(0, "order_id", "order_id"), "order_id"),
                JoinOutputColumn::right(cudf_query_engine::ColumnRef::input(1, "category", "category"), "category"),
            ])
            .with_build_side(JoinSide::Right),
        Arc::clone(&joined_schema),
    );
    let semi_keys =
        builder.add_source(SourceNode::LocalParquet(join_right_source(&semi_path, &semi_schema, &["order_id"])));
    let semi_join = builder.add_join(
        joined,
        semi_keys,
        JoinSpec::new(JoinType::LeftSemi, vec![join_order_id_key()])
            .with_output_projection(vec![
                JoinOutputColumn::left(cudf_query_engine::ColumnRef::input(0, "order_id", "order_id"), "order_id"),
                JoinOutputColumn::left(cudf_query_engine::ColumnRef::input(0, "category", "category"), "category"),
            ])
            .with_build_side(JoinSide::Right),
        Arc::clone(&joined_schema),
    );
    let aggregate = builder.add_aggregate(
        semi_join,
        AggregateSpec::new(
            vec![Expr::column("category")],
            vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("order_id"), "order_count")],
        ),
        Arc::clone(&aggregate_schema),
    );
    let plan = builder.finish_with_host_sink(aggregate);
    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 8 * 1024 * 1024 * 1024,
            max_source_chunk_bytes: 1,
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
    assert!(
        metrics
            .operators
            .iter()
            .filter_map(|operator| operator.join.as_ref())
            .any(|join| join.execution_strategy == "bounded_partitioned_hash_equi_join")
    );
    assert_eq!(metrics.filtered_join_build_count, 1);
    assert!(metrics.partial_aggregate_count > 1);
    assert!(
        metrics.final_merge_attempts > 1,
        "final_merge_attempts={}, partial_aggregate_count={}",
        metrics.final_merge_attempts,
        metrics.partial_aggregate_count
    );
    assert!(
        metrics.final_merge_levels > 1,
        "final_merge_levels={}, partial_aggregate_count={}",
        metrics.final_merge_levels,
        metrics.partial_aggregate_count
    );
    assert!(
        metrics.reservation_events.iter().all(|event| event.memory_boundary != "join_output_stream_concat"),
        "join streams feeding filtered join and aggregate should not concatenate at the join output boundary"
    );
    assert!(
        metrics
            .reservation_events
            .iter()
            .filter(|event| {
                event.outcome.as_str() == "granted" && event.memory_boundary == "compiled_aggregate_final_merge"
            })
            .count()
            > 1,
        "compiled aggregate stream final merge should use thresholded fan-in rather than one all-partials concat"
    );

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 2);
    let rows = string_array(batch, 0)
        .iter()
        .zip(int64_array(batch, 1).values())
        .map(|(category, count)| (category.expect("category should be non-null").to_string(), *count))
        .collect::<std::collections::BTreeMap<_, _>>();
    assert_eq!(rows.get("two"), Some(&2));
    assert_eq!(rows.get("three"), Some(&1));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_splits_join_leaf_source_read_after_reservation_denial() {
    let dir = unique_temp_dir("cudf-query-engine-bounded-root-hash-join-source-retry-split");
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
    let left = builder.add_source(SourceNode::LocalParquet(join_source_with_projected_row_group_estimates(
        &left_path,
        &left_schema,
        &["order_id", "qty"],
        4,
        1_000_000,
    )));
    let right = builder.add_source(SourceNode::LocalParquet(join_source_with_projected_row_group_estimates(
        &right_path,
        &right_schema,
        &["order_id", "category"],
        3,
        1_000_000,
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
        .with_shared_runtime(Arc::new(cudf::Runtime::new().unwrap()))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 3_500_000,
            max_source_chunk_bytes: 16 * 1024 * 1024,
            max_row_groups_per_chunk: 4,
            max_retry_attempts: 4,
            min_retry_chunk_row_groups: 1,
        })
        .build()
        .unwrap();

    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert!(metrics.retry_count >= 1);
    assert!(metrics.retry_reservation_failure_count >= 1);
    assert!(metrics.source_chunks_retried >= 1);
    assert!(
        metrics.reservation_events.iter().any(|event| {
            event.outcome.as_str() == "denied"
                && event.reservation_kind == ReservationKind::SourceRead
                && event.memory_boundary == "join_source_read"
                && event.requested_bytes == 4_000_000
        }),
        "coarse join source chunk should be denied before retry splitting"
    );
    assert!(
        metrics.reservation_events.iter().any(|event| {
            event.outcome.as_str() == "granted"
                && event.reservation_kind == ReservationKind::SourceRead
                && event.memory_boundary == "join_source_read"
                && event.requested_bytes == 2_000_000
        }),
        "split join source chunks should be admitted on retry attempts"
    );

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

fn join_source_with_projected_row_group_estimates(
    path: &Path,
    schema: &Arc<Schema>,
    columns: &[&str],
    row_group_count: usize,
    bytes_per_row_group: u64,
) -> LocalParquetSource {
    let row_group_bytes = vec![bytes_per_row_group; row_group_count];
    LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new(path)
                .with_file_size_bytes(fs::metadata(path).unwrap().len())
                .with_row_group_bytes(row_group_bytes.clone())
                .with_projected_row_group_bytes(row_group_bytes.clone(), row_group_bytes)
                .with_row_group_rows(vec![1; row_group_count]),
        ],
        Arc::clone(schema),
        columns.iter().map(|column| (*column).to_string()).collect(),
    )
}

#[test]
fn bounded_native_query_reports_join_leaf_source_read_hard_budget_failure() {
    let dir = unique_temp_dir("cudf-query-engine-bounded-join-source-hard-budget-failure");
    let left_path = dir.join("lineitem.parquet");
    let right_path = dir.join("orders.parquet");
    let left_schema = write_row_grouped_join_left_parquet(&left_path);
    let right_schema = write_row_grouped_join_right_parquet(&right_path);
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
        Field::new("category", DataType::Utf8, false),
    ]));
    let left_source = join_left_source(&left_path, &left_schema, &["order_id", "qty"]);
    let left_source_for_chunk_plan = left_source.clone();

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(left_source));
    let left_chunk_plan =
        plan_local_parquet_source_chunks(left, &left_source_for_chunk_plan, SourceChunkPlanningPolicy::new(None, 1))
            .expect("test source should produce row-group chunks");
    let admitted_budget = left_chunk_plan.chunks[0]
        .known_row_group_bytes
        .expect("test parquet metadata should expose row-group bytes")
        .saturating_add(1);
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
        output_schema,
    );
    let plan = builder.finish_with_host_sink(join);
    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::new(cudf::Runtime::new().unwrap()))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: admitted_budget,
            max_source_chunk_bytes: 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .build()
        .unwrap();

    let error = match execute_query_plan(&context, &plan) {
        Ok(_) => panic!("bounded join source read budget should fail"),
        Err(error) => error,
    };
    let admitted_budget = admitted_budget.to_string();
    assert_eq!(error.code(), QueryErrorCode::MemoryReservationFailed);
    assert_eq!(error.fact_value("budget_bytes"), Some(admitted_budget.as_str()));
    assert_eq!(error.fact_value("memory_boundary"), Some("join_source_read"));
    assert_eq!(error.fact_value("reservation_kind"), Some("source_read"));
    assert_eq!(error.fact_value("stage"), Some("join_source_read"));
    assert_eq!(error.fact_value("chunk_index"), Some("0"));
    assert_eq!(error.fact_value("file_index"), Some("0"));
    assert_eq!(error.fact_value("row_groups"), Some("0"));

    fs::remove_dir_all(dir).unwrap();
}
