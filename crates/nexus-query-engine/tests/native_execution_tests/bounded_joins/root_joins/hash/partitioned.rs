#[test]
fn bounded_native_query_routes_large_keyed_inner_build_to_partitioned_hash_join() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let dir = unique_temp_dir("nexus-query-engine-bounded-large-keyed-inner-partitioned-hash");
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
                JoinOutputColumn::left(nexus_query_engine::ColumnRef::input(0, "order_id", "order_id"), "order_id"),
                JoinOutputColumn::left(nexus_query_engine::ColumnRef::input(0, "qty", "qty"), "qty"),
                JoinOutputColumn::right(nexus_query_engine::ColumnRef::input(1, "category", "category"), "category"),
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
fn bounded_native_query_feeds_partitioned_root_join_stream_to_topk_without_concat() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let dir = unique_temp_dir("nexus-query-engine-bounded-root-hash-join-stream-topk");
    let left_path = dir.join("lineitem.parquet");
    let right_path = dir.join("orders.parquet");
    let left_schema = write_join_left_parquet(&runtime, &left_path);
    let right_schema = write_join_right_parquet(&runtime, &right_path);
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
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
                JoinOutputColumn::left(nexus_query_engine::ColumnRef::input(0, "order_id", "order_id"), "order_id"),
                JoinOutputColumn::left(nexus_query_engine::ColumnRef::input(0, "qty", "qty"), "qty"),
            ])
            .with_build_side(JoinSide::Right),
        Arc::clone(&output_schema),
    );
    let sort = builder.add_sort(
        join,
        SortSpec::new(vec![SortExpr::new(Expr::column("qty"), true, false)]),
        Arc::clone(&output_schema),
    );
    let limit = builder.add_limit(sort, LimitSpec::fetch(2), output_schema);
    let plan = builder.finish_with_host_sink(limit);
    let context = bounded_join_execution_context(&runtime);

    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    let join_metrics =
        metrics.operators.iter().find_map(|operator| operator.join.as_ref()).expect("join metrics should be recorded");
    assert_eq!(join_metrics.execution_strategy, "bounded_partitioned_hash_equi_join");
    assert!(
        metrics.reservation_events.iter().all(|event| event.memory_boundary != "join_output_stream_concat"),
        "root join stream feeding TopK should not materialize one concatenated join output"
    );
    assert!(metrics.reservation_events.iter().any(|event| {
        event.outcome.as_str() == "granted"
            && event.reservation_kind == ReservationKind::FinalSort
            && event.memory_boundary == "final_sort_topk_stream_chunk"
    }));
    assert!(metrics.budgeted_execution_events.iter().any(|event| {
        event.operation == "nexus_query_engine.final_sort.topk_prepare"
            && event.boundary == "final_sort_topk_stream_chunk"
    }));

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 2);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[3, 2]);
    assert_eq!(int32_array(batch, 1).values().as_ref(), &[30, 25]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_aliases_shared_partitioned_join_stream_for_probe_fanout() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let dir = unique_temp_dir("nexus-query-engine-bounded-shared-hash-join-stream");
    let fact_path = dir.join("fact.parquet");
    let prefix_dim_path = dir.join("prefix_dim.parquet");
    let first_dim_path = dir.join("first_dim.parquet");
    let second_dim_path = dir.join("second_dim.parquet");
    let fact_schema = write_join_left_parquet(&runtime, &fact_path);
    let prefix_dim_schema = write_join_right_parquet(&runtime, &prefix_dim_path);
    let first_dim_schema = write_join_right_parquet(&runtime, &first_dim_path);
    let second_dim_schema = write_join_right_parquet(&runtime, &second_dim_path);
    let prefix_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
    ]));
    let branch_schema = Arc::new(Schema::new(vec![
        Field::new("branch", DataType::Int32, false),
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
    ]));

    let mut prefix_dim = join_right_source(&prefix_dim_path, &prefix_dim_schema, &["order_id", "category"]);
    prefix_dim.files[0].row_group_rows = Some(vec![100_000_000]);

    let mut builder = QueryPlanBuilder::new();
    let fact =
        builder.add_source(SourceNode::LocalParquet(join_left_source(&fact_path, &fact_schema, &["order_id", "qty"])));
    let prefix_dim = builder.add_source(SourceNode::LocalParquet(prefix_dim));
    let prefix = builder.add_join(
        fact,
        prefix_dim,
        JoinSpec::new(JoinType::Inner, vec![join_order_id_key()])
            .with_output_projection(vec![
                JoinOutputColumn::left(nexus_query_engine::ColumnRef::input(0, "order_id", "order_id"), "order_id"),
                JoinOutputColumn::left(nexus_query_engine::ColumnRef::input(0, "qty", "qty"), "qty"),
            ])
            .with_build_side(JoinSide::Right),
        Arc::clone(&prefix_schema),
    );
    let prefix_projected = builder.add_projection(
        prefix,
        projection_outputs(vec![Expr::column("order_id"), Expr::column("qty")], Arc::clone(&prefix_schema)),
    );
    let first_dim = builder.add_source(SourceNode::LocalParquet(join_right_source(
        &first_dim_path,
        &first_dim_schema,
        &["order_id"],
    )));
    let second_dim = builder.add_source(SourceNode::LocalParquet(join_right_source(
        &second_dim_path,
        &second_dim_schema,
        &["order_id"],
    )));
    let first = builder.add_join(
        prefix_projected,
        first_dim,
        JoinSpec::new(JoinType::Inner, vec![join_order_id_key()])
            .with_output_projection(vec![
                JoinOutputColumn::left(nexus_query_engine::ColumnRef::input(0, "order_id", "order_id"), "order_id"),
                JoinOutputColumn::left(nexus_query_engine::ColumnRef::input(0, "qty", "qty"), "qty"),
            ])
            .with_build_side(JoinSide::Right),
        Arc::clone(&prefix_schema),
    );
    let second = builder.add_join(
        prefix_projected,
        second_dim,
        JoinSpec::new(JoinType::Inner, vec![join_order_id_key()])
            .with_output_projection(vec![
                JoinOutputColumn::left(nexus_query_engine::ColumnRef::input(0, "order_id", "order_id"), "order_id"),
                JoinOutputColumn::left(nexus_query_engine::ColumnRef::input(0, "qty", "qty"), "qty"),
            ])
            .with_build_side(JoinSide::Right),
        Arc::clone(&prefix_schema),
    );
    let first_projected = builder.add_projection(
        first,
        projection_outputs(
            vec![Expr::literal(LiteralValue::Int32(1)), Expr::column("order_id"), Expr::column("qty")],
            Arc::clone(&branch_schema),
        ),
    );
    let second_projected = builder.add_projection(
        second,
        projection_outputs(
            vec![Expr::literal(LiteralValue::Int32(2)), Expr::column("order_id"), Expr::column("qty")],
            Arc::clone(&branch_schema),
        ),
    );
    let union = builder.add_union(vec![first_projected, second_projected], Arc::clone(&branch_schema));
    let plan = builder.finish_with_host_sink(union);
    let context = bounded_join_execution_context(&runtime);

    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert!(
        metrics.reservation_events.iter().all(|event| {
            event.memory_boundary != "join_output_stream_concat" || event.node_id.as_deref() != Some("n2")
        }),
        "shared partitioned join output should remain chunked instead of concatenating for fan-out"
    );
    assert!(metrics.reservation_events.iter().any(|event| {
        event.outcome.as_str() == "applied"
            && event.action == MemoryReservationEventAction::AliasAcquire
            && event.memory_boundary == "compiled_join_probe_stream_alias"
    }));

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 6);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_streams_shared_reusable_join_output_for_probe_fanout() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let dir = unique_temp_dir("nexus-query-engine-bounded-shared-reusable-hash-join-stream");
    let fact_path = dir.join("fact.parquet");
    let prefix_dim_path = dir.join("prefix_dim.parquet");
    let first_dim_path = dir.join("first_dim.parquet");
    let second_dim_path = dir.join("second_dim.parquet");
    let fact_schema = write_join_left_parquet(&runtime, &fact_path);
    let prefix_dim_schema = write_join_right_parquet(&runtime, &prefix_dim_path);
    let first_dim_schema = write_join_right_parquet(&runtime, &first_dim_path);
    let second_dim_schema = write_join_right_parquet(&runtime, &second_dim_path);
    let prefix_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
    ]));
    let branch_schema = Arc::new(Schema::new(vec![
        Field::new("branch", DataType::Int32, false),
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let fact =
        builder.add_source(SourceNode::LocalParquet(join_left_source(&fact_path, &fact_schema, &["order_id", "qty"])));
    let prefix_dim = builder.add_source(SourceNode::LocalParquet(join_right_source(
        &prefix_dim_path,
        &prefix_dim_schema,
        &["order_id", "category"],
    )));
    let prefix = builder.add_join(
        fact,
        prefix_dim,
        JoinSpec::new(JoinType::Inner, vec![join_order_id_key()])
            .with_output_projection(vec![
                JoinOutputColumn::left(nexus_query_engine::ColumnRef::input(0, "order_id", "order_id"), "order_id"),
                JoinOutputColumn::left(nexus_query_engine::ColumnRef::input(0, "qty", "qty"), "qty"),
            ])
            .with_build_side(JoinSide::Right),
        Arc::clone(&prefix_schema),
    );
    let prefix_projected = builder.add_projection(
        prefix,
        projection_outputs(vec![Expr::column("order_id"), Expr::column("qty")], Arc::clone(&prefix_schema)),
    );
    let first_dim = builder.add_source(SourceNode::LocalParquet(join_right_source(
        &first_dim_path,
        &first_dim_schema,
        &["order_id"],
    )));
    let second_dim = builder.add_source(SourceNode::LocalParquet(join_right_source(
        &second_dim_path,
        &second_dim_schema,
        &["order_id"],
    )));
    let first = builder.add_join(
        prefix_projected,
        first_dim,
        JoinSpec::new(JoinType::Inner, vec![join_order_id_key()])
            .with_output_projection(vec![
                JoinOutputColumn::left(nexus_query_engine::ColumnRef::input(0, "order_id", "order_id"), "order_id"),
                JoinOutputColumn::left(nexus_query_engine::ColumnRef::input(0, "qty", "qty"), "qty"),
            ])
            .with_build_side(JoinSide::Right),
        Arc::clone(&prefix_schema),
    );
    let second = builder.add_join(
        prefix_projected,
        second_dim,
        JoinSpec::new(JoinType::Inner, vec![join_order_id_key()])
            .with_output_projection(vec![
                JoinOutputColumn::left(nexus_query_engine::ColumnRef::input(0, "order_id", "order_id"), "order_id"),
                JoinOutputColumn::left(nexus_query_engine::ColumnRef::input(0, "qty", "qty"), "qty"),
            ])
            .with_build_side(JoinSide::Right),
        Arc::clone(&prefix_schema),
    );
    let first_projected = builder.add_projection(
        first,
        projection_outputs(
            vec![Expr::literal(LiteralValue::Int32(1)), Expr::column("order_id"), Expr::column("qty")],
            Arc::clone(&branch_schema),
        ),
    );
    let second_projected = builder.add_projection(
        second,
        projection_outputs(
            vec![Expr::literal(LiteralValue::Int32(2)), Expr::column("order_id"), Expr::column("qty")],
            Arc::clone(&branch_schema),
        ),
    );
    let union = builder.add_union(vec![first_projected, second_projected], Arc::clone(&branch_schema));
    let plan = builder.finish_with_host_sink(union);
    let context = bounded_join_execution_context(&runtime);

    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert!(metrics.join_strategy_decisions.iter().any(|decision| {
        decision.node_id == "n2"
            && decision.facts.get("join_strategy").is_some_and(|strategy| strategy == "reusable_hash_equi")
    }));
    assert!(metrics.reservation_events.iter().any(|event| {
        event.outcome.as_str() == "granted"
            && event.node_id.as_deref() == Some("n2")
            && event.reservation_kind == ReservationKind::JoinOutput
            && event.memory_boundary == "join_output_stream_chunk"
    }));
    assert!(
        metrics
            .reservation_events
            .iter()
            .all(|event| { event.node_id.as_deref() != Some("n2") || event.memory_boundary != "join_output" }),
        "shared reusable join output should be chunked instead of materialized as one resident frame"
    );
    assert!(metrics.reservation_events.iter().any(|event| {
        event.outcome.as_str() == "applied"
            && event.action == MemoryReservationEventAction::AliasAcquire
            && event.memory_boundary == "compiled_join_probe_stream_alias"
    }));

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 6);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_routes_large_keyed_inner_residual_build_to_partitioned_mixed_join() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let dir = unique_temp_dir("nexus-query-engine-bounded-large-keyed-inner-residual-one-shot");
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
                JoinOutputColumn::left(nexus_query_engine::ColumnRef::input(0, "order_id", "order_id"), "order_id"),
                JoinOutputColumn::left(nexus_query_engine::ColumnRef::input(0, "qty", "qty"), "qty"),
                JoinOutputColumn::right(nexus_query_engine::ColumnRef::input(1, "category", "category"), "category"),
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
fn bounded_native_query_executes_or_residual_with_mixed_join() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let dir = unique_temp_dir("nexus-query-engine-bounded-or-residual-mixed-join");
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
    let residual = Expr::binary(
        Expr::binary(Expr::input_column(0, "qty", "qty"), BinaryOp::Lt, Expr::literal(LiteralValue::Int32(25))),
        BinaryOp::Or,
        Expr::binary(
            Expr::input_column(1, "category", "category"),
            BinaryOp::Eq,
            Expr::literal(LiteralValue::Utf8("three".to_string())),
        ),
    );
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(JoinType::Inner, vec![join_order_id_key()])
            .with_residual_predicate(residual)
            .with_output_projection(vec![
                JoinOutputColumn::left(nexus_query_engine::ColumnRef::input(0, "order_id", "order_id"), "order_id"),
                JoinOutputColumn::left(nexus_query_engine::ColumnRef::input(0, "qty", "qty"), "qty"),
                JoinOutputColumn::right(nexus_query_engine::ColumnRef::input(1, "category", "category"), "category"),
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
    let join_metrics =
        metrics.operators.iter().find_map(|operator| operator.join.as_ref()).expect("join metrics should be recorded");
    assert_eq!(join_metrics.execution_strategy, "bounded_mixed_keyed_residual_join");
    assert!(join_metrics.residual_predicate);
    assert_eq!(join_metrics.output_rows, 2);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 2);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[2, 3]);
    assert_eq!(int32_array(batch, 1).values().as_ref(), &[20, 30]);
    assert_eq!(string_array(batch, 2).iter().collect::<Vec<_>>(), vec![Some("two"), Some("three")]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_streams_partitioned_mixed_join_into_grouped_aggregate() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let dir = unique_temp_dir("nexus-query-engine-bounded-partitioned-mixed-stream-aggregate");
    let left_path = dir.join("lineitem.parquet");
    let right_path = dir.join("orders.parquet");
    let left_schema = write_join_left_parquet(&runtime, &left_path);
    let right_schema = write_join_right_parquet(&runtime, &right_path);
    let join_schema = Arc::new(Schema::new(vec![
        Field::new("qty", DataType::Int32, false),
        Field::new("category", DataType::Utf8, false),
    ]));
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("category", DataType::Utf8, false),
        Field::new("line_count", DataType::Int64, false),
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
                JoinOutputColumn::left(nexus_query_engine::ColumnRef::input(0, "qty", "qty"), "qty"),
                JoinOutputColumn::right(nexus_query_engine::ColumnRef::input(1, "category", "category"), "category"),
            ])
            .with_build_side(JoinSide::Right),
        Arc::clone(&join_schema),
    );
    let filtered = builder.add_filter(join, Expr::is_not_null(Expr::column("category")), Arc::clone(&join_schema));
    let projected = builder.add_projection(
        filtered,
        vec![
            ProjectionOutput::new(Expr::column("category"), Field::new("category", DataType::Utf8, false)),
            ProjectionOutput::new(Expr::column("qty"), Field::new("qty", DataType::Int32, false)),
        ],
    );
    let aggregate = builder.add_aggregate(
        projected,
        AggregateSpec::new(
            vec![Expr::column("category")],
            vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("qty"), "line_count")],
        ),
        Arc::clone(&aggregate_schema),
    );
    let plan = builder.finish_with_host_sink(aggregate);
    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 512 * 1024 * 1024,
            max_source_chunk_bytes: 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .build()
        .unwrap();

    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    let join_metrics =
        metrics.operators.iter().find_map(|operator| operator.join.as_ref()).expect("join metrics should be recorded");
    assert_eq!(join_metrics.execution_strategy, "bounded_partitioned_mixed_keyed_residual_join");
    assert_eq!(join_metrics.output_rows, 1);
    assert!(metrics.partial_aggregate_count > 0);
    assert!(metrics.join_strategy_decisions.iter().any(|decision| {
        decision.facts.get("join_strategy").map(String::as_str) == Some("partitioned_mixed_keyed_residual")
            && decision
                .facts
                .get("partition_count")
                .and_then(|value| value.parse::<usize>().ok())
                .is_some_and(|partition_count| partition_count > 1)
    }));
    assert!(metrics.reservation_events.iter().any(|event| {
        event.outcome.as_str() == "granted" && event.memory_boundary == "join_partition_mixed_size_data"
    }));
    assert!(metrics.reservation_events.iter().any(|event| {
        event.outcome.as_str() == "granted" && event.memory_boundary == "join_partition_output"
    }));
    assert!(metrics.reservation_events.iter().all(|event| {
        event.memory_boundary != "join_partition_mixed_size_data_forecast"
    }));
    let first_partial_work = metrics
        .reservation_events
        .iter()
        .position(|event| {
            event.outcome.as_str() == "granted"
                && event.action == MemoryReservationEventAction::Reserve
                && event.memory_boundary == "compiled_aggregate_partial_state_work"
        })
        .expect("aggregate should reserve partial work while consuming the join stream");
    let last_join_output = metrics
        .reservation_events
        .iter()
        .rposition(|event| {
            event.outcome.as_str() == "granted"
                && event.action == MemoryReservationEventAction::Reserve
                && event.memory_boundary == "join_partition_output"
        })
        .expect("partitioned mixed join should reserve partition outputs");
    assert!(
        first_partial_work < last_join_output,
        "aggregate partial work should begin before the final join partition output is reserved"
    );

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 1);
    assert_eq!(batch.num_columns(), 2);
    assert_eq!(string_array(batch, 0).iter().collect::<Vec<_>>(), vec![Some("two")]);
    assert_eq!(int64_array(batch, 1).values().as_ref(), &[1]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_streams_q72_shaped_partitioned_left_join_into_grouped_aggregate() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let dir = unique_temp_dir("nexus-query-engine-bounded-q72-left-stream-aggregate");
    let left_path = dir.join("catalog_sales.parquet");
    let right_path = dir.join("catalog_returns.parquet");
    let dim_path = dir.join("category_dim.parquet");
    let left_schema = write_join_left_parquet(&runtime, &left_path);
    let right_schema = write_join_right_keys_parquet(&runtime, &right_path, &[2, 2, 4]);
    let dim_schema = write_category_bucket_parquet(&runtime, &dim_path);
    let join_schema = Arc::new(Schema::new(vec![
        Field::new("qty", DataType::Int32, false),
        Field::new("category", DataType::Utf8, true),
    ]));
    let hash_join_schema = Arc::new(Schema::new(vec![
        Field::new("qty", DataType::Int32, false),
        Field::new("bucket", DataType::Int32, false),
    ]));
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("bucket", DataType::Int32, false),
        Field::new("line_count", DataType::Int64, false),
    ]));

    let mut right_source = join_right_source(&right_path, &right_schema, &["order_id", "category"]);
    right_source.files[0].row_group_rows = Some(vec![100_000_000]);

    let mut builder = QueryPlanBuilder::new();
    let left =
        builder.add_source(SourceNode::LocalParquet(join_left_source(&left_path, &left_schema, &["order_id", "qty"])));
    let right = builder.add_source(SourceNode::LocalParquet(right_source));
    let dim = builder.add_source(SourceNode::LocalParquet(join_right_source(
        &dim_path,
        &dim_schema,
        &["category", "bucket"],
    )));
    let left_join = builder.add_join(
        left,
        right,
        JoinSpec::new(JoinType::Left, vec![join_order_id_key()])
            .with_output_projection(vec![
                JoinOutputColumn::left(nexus_query_engine::ColumnRef::input(0, "qty", "qty"), "qty"),
                JoinOutputColumn::right(nexus_query_engine::ColumnRef::input(1, "category", "category"), "category"),
            ])
            .with_build_side(JoinSide::Right),
        Arc::clone(&join_schema),
    );
    let hash_join = builder.add_join(
        left_join,
        dim,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                nexus_query_engine::ColumnRef::input(0, "category", "category"),
                nexus_query_engine::ColumnRef::input(1, "category", "dim_category"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(nexus_query_engine::ColumnRef::input(0, "qty", "qty"), "qty"),
            JoinOutputColumn::right(nexus_query_engine::ColumnRef::input(1, "bucket", "bucket"), "bucket"),
        ])
        .with_build_side(JoinSide::Right),
        Arc::clone(&hash_join_schema),
    );
    let aggregate = builder.add_aggregate(
        hash_join,
        AggregateSpec::new(
            vec![Expr::column("bucket")],
            vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("qty"), "line_count")],
        ),
        Arc::clone(&aggregate_schema),
    );
    let plan = builder.finish_with_host_sink(aggregate);
    let context = bounded_join_execution_context(&runtime);

    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    let join_metrics =
        metrics.operators.iter().find_map(|operator| operator.join.as_ref()).expect("join metrics should be recorded");
    assert_eq!(join_metrics.join_type, "left");
    assert_eq!(join_metrics.execution_strategy, "bounded_partitioned_hash_equi_join");
    assert_eq!(join_metrics.output_rows, 6);
    assert_eq!(
        metrics.hash_join_build_count, 0,
        "partitioned hash joins should not build reusable hash-table artifacts"
    );
    assert!(metrics.partial_aggregate_count > 0);
    assert!(
        metrics
            .reservation_events
            .iter()
            .any(|event| { event.outcome.as_str() == "granted" && event.memory_boundary == "join_partition_output" })
    );
    assert!(
        metrics.reservation_events.iter().any(|event| {
            event.outcome.as_str() == "granted"
                && event.action == MemoryReservationEventAction::Reserve
                && event.memory_boundary == "compiled_aggregate_partial_state_work"
        }),
        "aggregate should consume streamed partitioned hash outputs frame by frame"
    );

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_columns(), 2);
    let mut rows = (0..batch.num_rows())
        .map(|row| (int32_array(batch, 0).value(row), int64_array(batch, 1).value(row)))
        .collect::<Vec<_>>();
    rows.sort();
    assert_eq!(rows, vec![(7, 4)]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_executes_keyed_inner_case_residual_as_one_shot_post_filter() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let dir = unique_temp_dir("nexus-query-engine-bounded-keyed-inner-case-residual-one-shot");
    let left_path = dir.join("lineitem.parquet");
    let right_path = dir.join("orders.parquet");
    let left_schema = write_join_left_parquet(&runtime, &left_path);
    let right_schema = write_join_right_parquet(&runtime, &right_path);
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
        Field::new("category", DataType::Utf8, false),
    ]));
    let residual = Expr::case(
        vec![nexus_query_engine::CaseWhen::new(
            qty_gt_right_order_id_times_ten(),
            Expr::literal(LiteralValue::Boolean(true)),
        )],
        Expr::literal(LiteralValue::Boolean(false)),
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
                JoinOutputColumn::left(nexus_query_engine::ColumnRef::input(0, "order_id", "order_id"), "order_id"),
                JoinOutputColumn::left(nexus_query_engine::ColumnRef::input(0, "qty", "qty"), "qty"),
                JoinOutputColumn::right(nexus_query_engine::ColumnRef::input(1, "category", "category"), "category"),
            ])
            .with_build_side(JoinSide::Right),
        Arc::clone(&output_schema),
    );
    let plan = builder.finish_with_host_sink(join);
    assert!(plan.capability_analysis().admits_root_join());
    let context = bounded_join_execution_context(&runtime);

    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.hash_join_build_count, 0);
    assert_eq!(metrics.one_shot_join_count, 1);
    let join_metrics =
        metrics.operators.iter().find_map(|operator| operator.join.as_ref()).expect("join metrics should be recorded");
    assert_eq!(join_metrics.execution_strategy, "bounded_one_shot_equi_join");
    assert!(join_metrics.residual_predicate);
    assert_eq!(join_metrics.output_rows, 1);
    assert!(metrics.join_strategy_decisions.iter().any(|decision| {
        decision.facts.get("join_strategy").map(String::as_str) == Some("one_shot_equi")
            && decision.facts.values().any(|value| value == "residual_requires_post_join_filter")
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
    let dir = unique_temp_dir("nexus-query-engine-bounded-large-keyed-left-partitioned-hash");
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
                JoinOutputColumn::left(nexus_query_engine::ColumnRef::input(0, "order_id", "order_id"), "order_id"),
                JoinOutputColumn::left(nexus_query_engine::ColumnRef::input(0, "qty", "qty"), "qty"),
                JoinOutputColumn::right(nexus_query_engine::ColumnRef::input(1, "category", "category"), "category"),
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
    let dir = unique_temp_dir("nexus-query-engine-bounded-large-keyed-full-partitioned-hash");
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
                    nexus_query_engine::ColumnRef::input(0, "order_id", "left_order_id"),
                    "left_order_id",
                ),
                JoinOutputColumn::right(
                    nexus_query_engine::ColumnRef::input(1, "order_id", "right_order_id"),
                    "right_order_id",
                ),
                JoinOutputColumn::left(nexus_query_engine::ColumnRef::input(0, "qty", "qty"), "qty"),
                JoinOutputColumn::right(nexus_query_engine::ColumnRef::input(1, "category", "category"), "category"),
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
fn bounded_native_query_consumes_partitioned_join_stream_directly_in_grouped_aggregate() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let dir = unique_temp_dir("nexus-query-engine-bounded-direct-aggregate-stream");
    let left_path = dir.join("lineitem.parquet");
    let right_path = dir.join("orders.parquet");
    let left_schema = write_row_grouped_join_left_parquet(&left_path);
    let right_schema = write_join_right_parquet(&runtime, &right_path);
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
                JoinOutputColumn::left(nexus_query_engine::ColumnRef::input(0, "order_id", "order_id"), "order_id"),
                JoinOutputColumn::right(nexus_query_engine::ColumnRef::input(1, "category", "category"), "category"),
            ])
            .with_build_side(JoinSide::Right),
        Arc::clone(&joined_schema),
    );
    let aggregate = builder.add_aggregate(
        joined,
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
    assert!(
        metrics
            .operators
            .iter()
            .filter_map(|operator| operator.join.as_ref())
            .any(|join| join.execution_strategy == "bounded_partitioned_hash_equi_join")
    );
    assert!(
        metrics.partial_aggregate_count > 2,
        "direct join-output stream should produce one partial aggregate per streamed join batch"
    );
    assert!(
        metrics.partial_aggregate_input_rows >= metrics.partial_aggregate_rows,
        "partial aggregate metrics should distinguish input rows from retained state rows"
    );
    assert!(
        metrics.final_merge_attempts > 1,
        "direct join-output aggregate should use thresholded final-merge fan-in, attempts={}, partials={}",
        metrics.final_merge_attempts,
        metrics.partial_aggregate_count
    );
    assert!(
        metrics.final_merge_levels > 1,
        "direct join-output aggregate should merge partial states level by level, levels={}, partials={}",
        metrics.final_merge_levels,
        metrics.partial_aggregate_count
    );
    assert!(
        metrics.final_merge_max_fan_in < metrics.partial_aggregate_count,
        "direct join-output aggregate should not use a single all-partials final merge"
    );
    assert!(
        metrics.partial_aggregate_count <= metrics.source_chunks_committed.saturating_mul(2),
        "direct join-output aggregate should keep stream partial count proportional to source chunks, partials={}, source_chunks={}",
        metrics.partial_aggregate_count,
        metrics.source_chunks_committed
    );
    assert!(
        metrics.reservation_events.iter().all(|event| event.memory_boundary != "compiled_aggregate"),
        "direct join-output stream should not fall back to one full-frame compiled aggregate"
    );
    assert!(
        metrics.reservation_events.iter().all(|event| event.memory_boundary != "join_output_stream_concat"),
        "direct join-output stream should feed grouped aggregate without a full join-output concat"
    );
    assert!(
        metrics.reservation_events.iter().any(|event| {
            event.action == MemoryReservationEventAction::Transfer
                && event.reservation_kind == ReservationKind::PartialState
                && event.memory_boundary == "compiled_aggregate_partial_state"
        }),
        "direct stream aggregate should retain per-batch partial aggregate states"
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
fn bounded_native_query_preserves_partitioned_join_stream_through_filtered_join_into_aggregate() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let dir = unique_temp_dir("nexus-query-engine-bounded-one-shot-filtered-aggregate-stream");
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
                JoinOutputColumn::left(nexus_query_engine::ColumnRef::input(0, "order_id", "order_id"), "order_id"),
                JoinOutputColumn::right(nexus_query_engine::ColumnRef::input(1, "category", "category"), "category"),
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
                JoinOutputColumn::left(nexus_query_engine::ColumnRef::input(0, "order_id", "order_id"), "order_id"),
                JoinOutputColumn::left(nexus_query_engine::ColumnRef::input(0, "category", "category"), "category"),
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
    assert!(
        metrics.partial_aggregate_count > 1,
        "filtered direct join-output aggregate should produce multiple streamed partials"
    );
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
        metrics.partial_aggregate_count <= metrics.source_chunks_committed.saturating_mul(2),
        "filtered direct join-output aggregate should keep stream partial count proportional to source chunks, partials={}, source_chunks={}",
        metrics.partial_aggregate_count,
        metrics.source_chunks_committed
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
