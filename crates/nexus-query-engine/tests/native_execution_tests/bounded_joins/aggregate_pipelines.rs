use super::*;

#[test]
fn bounded_native_query_executes_root_hash_join_under_aggregate_with_ordered_reservations() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let dir = unique_temp_dir("nexus-query-engine-bounded-root-join-aggregate");
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
                JoinOutputColumn::left(nexus_query_engine::ColumnRef::input(0, "order_id", "order_id"), "order_id"),
                JoinOutputColumn::left(nexus_query_engine::ColumnRef::input(0, "qty", "qty"), "qty"),
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
    assert_eq!(metrics.compiled_pipeline_attempts, 1);
    assert!(metrics.compiled_pipeline_tasks_executed > 0);
    assert_eq!(metrics.compiled_pipeline_host_arrow_sinks, 1);
    assert_eq!(metrics.partial_aggregate_count, 1);
    assert_eq!(metrics.final_merge_attempts, 0);
    assert_eq!(metrics.boundary.device_to_host_materializations, 1);
    let join_metrics =
        metrics.operators.iter().find_map(|operator| operator.join.as_ref()).expect("join metrics should be recorded");
    assert_eq!(join_metrics.join_type, "inner");
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
            (ReservationKind::GatherMap, "join_gather_map_stream_slice"),
            (ReservationKind::JoinOutput, "join_output_stream_chunk"),
            (ReservationKind::PartialState, "compiled_aggregate_partial_state_work"),
            (ReservationKind::PartialState, "compiled_aggregate_partial_state"),
            (ReservationKind::FinalMerge, "compiled_aggregate_final_projection"),
            (ReservationKind::OperatorOutput, "compiled_aggregate_output"),
            (ReservationKind::FinalSort, "final_sort"),
            (ReservationKind::FinalSort, "final_sort"),
            (ReservationKind::Materialization, "host_materialization"),
        ],
    );
    assert!(
        metrics
            .reservation_events
            .iter()
            .all(|event| event.memory_boundary != "compiled_limit" && event.memory_boundary != "compiled_limit_output"),
        "compiled Sort+Limit should execute as a fused final-sort boundary"
    );

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
fn bounded_native_query_fuses_compiled_sort_limit_with_topk() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let dir = unique_temp_dir("nexus-query-engine-bounded-compiled-sort-limit-topk");
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
                JoinOutputColumn::left(nexus_query_engine::ColumnRef::input(0, "order_id", "order_id"), "order_id"),
                JoinOutputColumn::left(nexus_query_engine::ColumnRef::input(0, "qty", "qty"), "qty"),
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
    let sort = builder.add_sort(
        aggregate,
        SortSpec::new(vec![
            SortExpr::new(Expr::column("line_count"), true, false),
            SortExpr::new(Expr::column("order_id"), false, false),
        ]),
        Arc::clone(&aggregate_schema),
    );
    let limit = builder.add_limit(sort, LimitSpec::fetch(1), aggregate_schema);
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
    assert!(
        metrics
            .budgeted_execution_events
            .iter()
            .any(|event| event.operation == "nexus_query_engine.final_sort.topk_prepare"
                && event.boundary == "final_sort_topk_chunk"),
        "selective compiled Sort+Limit should use top-k as the normal final-sort path"
    );
    assert!(
        metrics
            .budgeted_execution_events
            .iter()
            .all(|event| event.operation != "nexus_query_engine.final_sort.prepare"),
        "selective compiled Sort+Limit should not full-sort before top-k"
    );
    assert!(
        metrics
            .reservation_events
            .iter()
            .all(|event| event.memory_boundary != "compiled_limit" && event.memory_boundary != "compiled_limit_output"),
        "compiled Limit should pass through the fused final-sort output"
    );

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 1);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[2]);
    assert_eq!(int64_array(batch, 1).values().as_ref(), &[2]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_executes_serial_aggregate_join_spine() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let dir = unique_temp_dir("nexus-query-engine-bounded-serial-aggregate-spine");
    let customer_path = dir.join("customer.parquet");
    let orders_path = dir.join("orders.parquet");
    let customer_schema = write_customer_order_count_customer_parquet(&runtime, &customer_path);
    let orders_schema = write_customer_order_count_orders_parquet(&runtime, &orders_path);
    let join_schema = Arc::new(Schema::new(vec![
        Field::new("c_custkey", DataType::Int32, false),
        Field::new("o_orderkey", DataType::Int32, true),
    ]));
    let customer_count_schema = Arc::new(Schema::new(vec![
        Field::new("c_custkey", DataType::Int32, false),
        Field::new("c_count", DataType::Int64, false),
    ]));
    let c_count_schema = Arc::new(Schema::new(vec![Field::new("c_count", DataType::Int64, false)]));
    let final_schema = Arc::new(Schema::new(vec![
        Field::new("c_count", DataType::Int64, false),
        Field::new("custdist", DataType::Int64, false),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let customers = builder.add_source(SourceNode::LocalParquet(join_left_source(
        &customer_path,
        &customer_schema,
        &["c_custkey"],
    )));
    let orders = builder.add_source(SourceNode::LocalParquet(join_right_source(
        &orders_path,
        &orders_schema,
        &["o_orderkey", "o_custkey"],
    )));
    let join = builder.add_join(
        customers,
        orders,
        JoinSpec::new(
            JoinType::Left,
            vec![JoinKey::new(
                nexus_query_engine::ColumnRef::input(0, "c_custkey", "c_custkey"),
                nexus_query_engine::ColumnRef::input(1, "o_custkey", "o_custkey"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(nexus_query_engine::ColumnRef::input(0, "c_custkey", "c_custkey"), "c_custkey"),
            JoinOutputColumn::right(nexus_query_engine::ColumnRef::input(1, "o_orderkey", "o_orderkey"), "o_orderkey"),
        ])
        .with_build_side(JoinSide::Right),
        Arc::clone(&join_schema),
    );
    let projection = builder.add_projection(
        join,
        projection_outputs(vec![Expr::column("c_custkey"), Expr::column("o_orderkey")], Arc::clone(&join_schema)),
    );
    let customer_counts = builder.add_aggregate(
        projection,
        AggregateSpec::new(
            vec![Expr::column("c_custkey")],
            vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("o_orderkey"), "c_count")],
        ),
        customer_count_schema,
    );
    let count_projection =
        builder.add_projection(customer_counts, projection_outputs(vec![Expr::column("c_count")], c_count_schema));
    let distribution = builder.add_aggregate(
        count_projection,
        AggregateSpec::new(
            vec![Expr::column("c_count")],
            vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("c_count"), "custdist")],
        ),
        Arc::clone(&final_schema),
    );
    let sort = builder.add_sort(
        distribution,
        SortSpec::new(vec![
            SortExpr::new(Expr::column("custdist"), true, false),
            SortExpr::new(Expr::column("c_count"), true, false),
        ]),
        final_schema,
    );
    let plan = builder.finish_with_host_sink(sort);
    assert!(!plan.capability_analysis().admits_root_join());
    assert!(plan.capability_analysis().admits_single_join_post_root_pipeline());
    assert!(!plan.capability_analysis().admits_closed_multi_join_subgraph());

    let context = bounded_join_execution_context(&runtime);
    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.phase_c_memory_control_policy.as_str(), "bounded");
    assert_eq!(metrics.reservation_denied_count, 0);
    assert_eq!(metrics.hash_join_build_count, 1);
    assert_eq!(metrics.compiled_pipeline_attempts, 1);
    assert!(metrics.compiled_pipeline_tasks_executed > 0);
    assert_eq!(metrics.compiled_pipeline_host_arrow_sinks, 1);
    assert_eq!(metrics.boundary.device_to_host_materializations, 1);
    let join_metrics =
        metrics.operators.iter().find_map(|operator| operator.join.as_ref()).expect("join metrics should be recorded");
    assert_eq!(join_metrics.join_type, "left");
    assert_eq!(join_metrics.output_rows, 5);
    assert_eq!(
        metrics
            .reservation_events
            .iter()
            .filter(|event| event.outcome.as_str() == "granted" && event.memory_boundary == "compiled_aggregate")
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
                    && event.memory_boundary == "compiled_aggregate_output"
            })
            .count(),
        1
    );
    assert_granted_reservation_trace(
        metrics,
        &[
            (ReservationKind::SourceRead, "join_source_read"),
            (ReservationKind::SourceRead, "join_source_read"),
            (ReservationKind::BuildSide, "join_build_side"),
            (ReservationKind::HashTable, "join_hash_table"),
            (ReservationKind::ProbeSide, "join_probe_side"),
            (ReservationKind::GatherMap, "join_gather_map"),
            (ReservationKind::GatherMap, "join_gather_map_stream_slice"),
            (ReservationKind::JoinOutput, "join_output_stream_chunk"),
            (ReservationKind::OperatorOutput, "compiled_projection"),
            (ReservationKind::OperatorOutput, "compiled_projection_output"),
            (ReservationKind::PartialState, "compiled_aggregate_partial_state_work"),
            (ReservationKind::PartialState, "compiled_aggregate_partial_state"),
            (ReservationKind::FinalMerge, "compiled_aggregate_final_projection"),
            (ReservationKind::OperatorOutput, "compiled_aggregate_output"),
            (ReservationKind::OperatorOutput, "compiled_projection"),
            (ReservationKind::OperatorOutput, "compiled_projection_output"),
            (ReservationKind::OperatorOutput, "compiled_aggregate"),
            (ReservationKind::OperatorOutput, "compiled_aggregate_output"),
            (ReservationKind::OperatorOutput, "compiled_aggregate_output"),
            (ReservationKind::FinalSort, "final_sort"),
            (ReservationKind::FinalSort, "final_sort"),
            (ReservationKind::Materialization, "host_materialization"),
        ],
    );

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 3);
    assert_eq!(batch.num_columns(), 2);
    assert_eq!(int64_array(batch, 0).values().as_ref(), &[0, 2, 1]);
    assert_eq!(int64_array(batch, 1).values().as_ref(), &[2, 1, 1]);

    fs::remove_dir_all(dir).unwrap();
}
