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
            (ReservationKind::ChunkOperator, "join_output_string_length_sizing"),
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
fn bounded_native_query_feeds_partitioned_root_join_stream_to_topk_without_concat() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let dir = unique_temp_dir("cudf-query-engine-bounded-root-hash-join-stream-topk");
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
                JoinOutputColumn::left(cudf_query_engine::ColumnRef::input(0, "order_id", "order_id"), "order_id"),
                JoinOutputColumn::left(cudf_query_engine::ColumnRef::input(0, "qty", "qty"), "qty"),
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
        event.operation == "cudf_query_engine.final_sort.topk_prepare"
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
    let dir = unique_temp_dir("cudf-query-engine-bounded-shared-hash-join-stream");
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
                JoinOutputColumn::left(cudf_query_engine::ColumnRef::input(0, "order_id", "order_id"), "order_id"),
                JoinOutputColumn::left(cudf_query_engine::ColumnRef::input(0, "qty", "qty"), "qty"),
            ])
            .with_build_side(JoinSide::Right),
        Arc::clone(&prefix_schema),
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
        prefix,
        first_dim,
        JoinSpec::new(JoinType::Inner, vec![join_order_id_key()])
            .with_output_projection(vec![
                JoinOutputColumn::left(cudf_query_engine::ColumnRef::input(0, "order_id", "order_id"), "order_id"),
                JoinOutputColumn::left(cudf_query_engine::ColumnRef::input(0, "qty", "qty"), "qty"),
            ])
            .with_build_side(JoinSide::Right),
        Arc::clone(&prefix_schema),
    );
    let second = builder.add_join(
        prefix,
        second_dim,
        JoinSpec::new(JoinType::Inner, vec![join_order_id_key()])
            .with_output_projection(vec![
                JoinOutputColumn::left(cudf_query_engine::ColumnRef::input(0, "order_id", "order_id"), "order_id"),
                JoinOutputColumn::left(cudf_query_engine::ColumnRef::input(0, "qty", "qty"), "qty"),
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
fn bounded_native_query_streams_partitioned_mixed_join_into_grouped_aggregate() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let dir = unique_temp_dir("cudf-query-engine-bounded-partitioned-mixed-stream-aggregate");
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
                JoinOutputColumn::left(cudf_query_engine::ColumnRef::input(0, "qty", "qty"), "qty"),
                JoinOutputColumn::right(cudf_query_engine::ColumnRef::input(1, "category", "category"), "category"),
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
    let context = bounded_join_execution_context(&runtime);

    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    let join_metrics =
        metrics.operators.iter().find_map(|operator| operator.join.as_ref()).expect("join metrics should be recorded");
    assert_eq!(join_metrics.execution_strategy, "bounded_partitioned_mixed_keyed_residual_join");
    assert_eq!(join_metrics.output_rows, 1);
    assert!(metrics.partial_aggregate_count > 0);
    assert!(metrics.reservation_events.iter().any(|event| {
        event.outcome.as_str() == "granted" && event.memory_boundary == "join_partition_mixed_size_data_forecast"
    }));

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
    let dir = unique_temp_dir("cudf-query-engine-bounded-q72-left-stream-aggregate");
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
                JoinOutputColumn::left(cudf_query_engine::ColumnRef::input(0, "qty", "qty"), "qty"),
                JoinOutputColumn::right(cudf_query_engine::ColumnRef::input(1, "category", "category"), "category"),
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
                cudf_query_engine::ColumnRef::input(0, "category", "category"),
                cudf_query_engine::ColumnRef::input(1, "category", "dim_category"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(cudf_query_engine::ColumnRef::input(0, "qty", "qty"), "qty"),
            JoinOutputColumn::right(cudf_query_engine::ColumnRef::input(1, "bucket", "bucket"), "bucket"),
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
    let dir = unique_temp_dir("cudf-query-engine-bounded-keyed-inner-case-residual-one-shot");
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
        vec![cudf_query_engine::CaseWhen::new(
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
                JoinOutputColumn::left(cudf_query_engine::ColumnRef::input(0, "order_id", "order_id"), "order_id"),
                JoinOutputColumn::left(cudf_query_engine::ColumnRef::input(0, "qty", "qty"), "qty"),
                JoinOutputColumn::right(cudf_query_engine::ColumnRef::input(1, "category", "category"), "category"),
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
fn bounded_native_query_reconciles_streamed_join_probe_prefetch_after_decode() {
    let dir = unique_temp_dir("cudf-query-engine-bounded-root-hash-join-prefetch-reconcile");
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
        .with_source_scheduler_policy(SourceSchedulerPolicy::new(2))
        .build()
        .unwrap();

    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.source_chunks_planned, 7);
    assert_eq!(metrics.source_chunks_committed, 7);
    assert_eq!(metrics.source_read_prefetches_started, metrics.source_read_prefetches_completed);
    assert!(metrics.source_read_prefetches_completed > 0);
    assert_eq!(metrics.source_read_prefetches_abandoned, 0);
    assert_eq!(metrics.reservation_denied_count, 0);
    assert_eq!(metrics.retry_resource_exhausted_count, 0);
    assert!(metrics.reservation_events.iter().any(|event| {
        event.outcome.as_str() == "granted"
            && event.action == MemoryReservationEventAction::Reserve
            && event.reservation_kind == ReservationKind::SourceRead
            && event.memory_boundary == "join_source_read_prefetch"
    }));
    let first_join_prefetch_reserve = metrics
        .reservation_events
        .iter()
        .position(|event| {
            event.outcome.as_str() == "granted"
                && event.action == MemoryReservationEventAction::Reserve
                && event.reservation_kind == ReservationKind::SourceRead
                && event.memory_boundary == "join_source_read_prefetch"
        })
        .expect("join probe prefetch should reserve its initial estimate");
    assert!(metrics.reservation_events[first_join_prefetch_reserve..].iter().any(|event| {
        event.outcome.as_str() == "granted"
            && event.action == MemoryReservationEventAction::Grow
            && event.reservation_kind == ReservationKind::SourceRead
            && event.memory_boundary == "join_source_read"
            && event.source_read_contract_quality.as_deref() == Some("observed_grow_shrink")
    }));

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
    let dir = unique_temp_dir("cudf-query-engine-bounded-root-hash-join-prefetch-no-headroom");
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
    assert!(metrics.source_read_prefetches_skipped > 0);
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
fn bounded_native_query_executes_q19_shaped_join_source_filter_predicate() {
    let dir = unique_temp_dir("cudf-query-engine-bounded-root-hash-join-q19-source-filter");
    let left_path = dir.join("lineitem_part.parquet");
    let right_path = dir.join("part_category.parquet");
    let left_schema = write_q19_shaped_join_source_left_parquet(&left_path);
    let right_schema = write_q19_shaped_join_source_right_parquet(&right_path);
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
    let filtered_left = builder.add_filter(left, q19_shaped_predicate(), Arc::clone(&left_schema));
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
                cudf_query_engine::ColumnRef::input(0, "partkey", "left_partkey"),
                cudf_query_engine::ColumnRef::input(1, "partkey", "right_partkey"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(cudf_query_engine::ColumnRef::input(0, "partkey", "partkey"), "partkey"),
            JoinOutputColumn::right(cudf_query_engine::ColumnRef::input(1, "category", "category"), "category"),
        ])
        .with_build_side(JoinSide::Right),
        Arc::clone(&output_schema),
    );
    let plan = builder.finish_with_host_sink(join);
    assert!(
        plan.capability_analysis().admits_strict_bounded_join(),
        "Q19-shaped source-adjacent filter should stay capability-admitted:\n{plan}"
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
        "Q19-shaped filter must execute on the join_source_filter reservation boundary"
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
fn bounded_native_query_consumes_partitioned_join_stream_directly_in_grouped_aggregate() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let dir = unique_temp_dir("cudf-query-engine-bounded-direct-aggregate-stream");
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
                JoinOutputColumn::left(cudf_query_engine::ColumnRef::input(0, "order_id", "order_id"), "order_id"),
                JoinOutputColumn::right(cudf_query_engine::ColumnRef::input(1, "category", "category"), "category"),
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
            device_budget_bytes: 2_500_000,
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
                && event.requested_bytes == 3_000_000
        }),
        "coarse join source chunk should be denied before retry splitting"
    );
    assert!(
        metrics.reservation_events.iter().any(|event| {
            event.outcome.as_str() == "granted"
                && event.reservation_kind == ReservationKind::SourceRead
                && event.memory_boundary == "join_source_read"
                && matches!(event.requested_bytes, 1_000_000 | 2_000_000)
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

fn write_row_grouped_join_left_with_payload_parquet(path: &Path) -> Arc<Schema> {
    let row_groups = 4;
    let rows_per_row_group = 256;
    let total_rows = row_groups * rows_per_row_group;
    let schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
        Field::new("payload", DataType::Utf8, false),
    ]));
    let mut order_ids = Vec::with_capacity(total_rows);
    let mut quantities = Vec::with_capacity(total_rows);
    for row_group in 0..row_groups {
        let order_id = i32::try_from(row_group + 1).unwrap();
        order_ids.extend(std::iter::repeat_n(order_id, rows_per_row_group));
        quantities.extend((0..rows_per_row_group).map(|row| order_id * 1000 + i32::try_from(row).unwrap()));
    }
    let payload = "x".repeat(8192);
    let payloads = vec![payload.as_str(); total_rows];
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(order_ids)),
            Arc::new(Int32Array::from(quantities)),
            Arc::new(StringArray::from(payloads)),
        ],
    )
    .unwrap();
    write_record_batch_with_row_group_rows(path, Arc::clone(&schema), batch, rows_per_row_group);
    schema
}

fn write_category_bucket_parquet(runtime: &cudf::Runtime, path: &Path) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("category", DataType::Utf8, false),
        Field::new("bucket", DataType::Int32, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(StringArray::from(vec!["category_2", "category_4"])), Arc::new(Int32Array::from(vec![7, 9]))],
    )
    .unwrap();
    DataFrame::from_record_batch(runtime, &batch).unwrap().write_parquet(path).unwrap();
    schema
}

fn write_q19_shaped_join_source_left_parquet(path: &Path) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("partkey", DataType::Int32, false),
        Field::new("brand", DataType::Utf8, false),
        Field::new("container", DataType::Utf8, false),
        Field::new("quantity", DataType::Float64, false),
        Field::new("size", DataType::Int32, false),
        Field::new("shipmode", DataType::Utf8, false),
        Field::new("shipinstruct", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8])),
            Arc::new(StringArray::from(vec![
                "Brand#12", "Brand#23", "Brand#34", "Brand#12", "Brand#12", "Brand#23", "Brand#34", "Brand#34",
            ])),
            Arc::new(StringArray::from(vec![
                "SM CASE", "MED BOX", "LG PACK", "SM CASE", "WRAP", "MED BOX", "LG PACK", "LG PACK",
            ])),
            Arc::new(Float64Array::from(vec![5.0, 15.0, 25.0, 12.0, 5.0, 9.0, 25.0, 25.0])),
            Arc::new(Int32Array::from(vec![3, 8, 12, 3, 3, 8, 12, 12])),
            Arc::new(StringArray::from(vec!["AIR", "AIR REG", "AIR", "AIR", "AIR", "AIR", "SHIP", "AIR"])),
            Arc::new(StringArray::from(vec![
                "DELIVER IN PERSON",
                "DELIVER IN PERSON",
                "DELIVER IN PERSON",
                "DELIVER IN PERSON",
                "DELIVER IN PERSON",
                "DELIVER IN PERSON",
                "DELIVER IN PERSON",
                "TAKE BACK RETURN",
            ])),
        ],
    )
    .unwrap();
    write_record_batch_with_row_group_rows(path, Arc::clone(&schema), batch, 2);
    schema
}

fn write_q19_shaped_join_source_right_parquet(path: &Path) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("partkey", DataType::Int32, false),
        Field::new("category", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8])),
            Arc::new(StringArray::from(vec![
                "brand12",
                "brand23",
                "brand34",
                "qty_high",
                "container",
                "qty_low",
                "mode",
                "instruction",
            ])),
        ],
    )
    .unwrap();
    write_record_batch_with_row_group_rows(path, Arc::clone(&schema), batch, 8);
    schema
}

fn q19_shaped_predicate() -> Expr {
    let branch_1 = q19_brand_branch("Brand#12", &["SM CASE", "SM BOX", "SM PACK", "SM PKG"], 1.0, 11.0, 5);
    let branch_2 = q19_brand_branch("Brand#23", &["MED BAG", "MED BOX", "MED PKG", "MED PACK"], 10.0, 20.0, 10);
    let branch_3 = q19_brand_branch("Brand#34", &["LG CASE", "LG BOX", "LG PACK", "LG PKG"], 20.0, 30.0, 15);
    Expr::binary(Expr::binary(branch_1, BinaryOp::Or, branch_2), BinaryOp::Or, branch_3)
}

fn q19_brand_branch(brand: &str, containers: &[&str], quantity_min: f64, quantity_max: f64, size_max: i32) -> Expr {
    and_all(vec![
        Expr::binary(Expr::column("brand"), BinaryOp::Eq, utf8_lit(brand)),
        Expr::in_list(Expr::column("container"), containers.iter().map(|container| utf8_lit(container)).collect()),
        Expr::binary(Expr::column("quantity"), BinaryOp::GtEq, Expr::literal(LiteralValue::Float64(quantity_min))),
        Expr::binary(Expr::column("quantity"), BinaryOp::LtEq, Expr::literal(LiteralValue::Float64(quantity_max))),
        Expr::binary(Expr::column("size"), BinaryOp::GtEq, Expr::literal(LiteralValue::Int32(1))),
        Expr::binary(Expr::column("size"), BinaryOp::LtEq, Expr::literal(LiteralValue::Int32(size_max))),
        Expr::in_list(Expr::column("shipmode"), vec![utf8_lit("AIR"), utf8_lit("AIR REG")]),
        Expr::binary(Expr::column("shipinstruct"), BinaryOp::Eq, utf8_lit("DELIVER IN PERSON")),
    ])
}

fn and_all(mut predicates: Vec<Expr>) -> Expr {
    let first = predicates.remove(0);
    predicates.into_iter().fold(first, |left, right| Expr::binary(left, BinaryOp::And, right))
}

fn utf8_lit(value: &str) -> Expr {
    Expr::literal(LiteralValue::Utf8(value.to_string()))
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
