use super::*;

#[test]
fn bounded_native_query_executes_pure_hash_multi_join_pipeline_with_ordered_reservations() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let dir = unique_temp_dir("nexus-query-engine-bounded-pure-hash-multi-join");
    let left_path = dir.join("lineitem.parquet");
    let orders_path = dir.join("orders.parquet");
    let region_path = dir.join("regions.parquet");
    let left_schema = write_join_left_parquet(&runtime, &left_path);
    let orders_schema = write_join_right_parquet(&runtime, &orders_path);
    let region_schema = write_join_region_parquet(&runtime, &region_path);
    let first_join_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
        Field::new("category", DataType::Utf8, false),
    ]));
    let second_join_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
        Field::new("region", DataType::Utf8, false),
    ]));
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("region", DataType::Utf8, false),
        Field::new("line_count", DataType::Int64, false),
    ]));
    let sort = SortSpec::new(vec![SortExpr::new(Expr::column("region"), false, false)]);

    let mut builder = QueryPlanBuilder::new();
    let lineitem =
        builder.add_source(SourceNode::LocalParquet(join_left_source(&left_path, &left_schema, &["order_id", "qty"])));
    let orders = builder.add_source(SourceNode::LocalParquet(join_right_source(
        &orders_path,
        &orders_schema,
        &["order_id", "category"],
    )));
    let regions = builder.add_source(SourceNode::LocalParquet(join_right_source(
        &region_path,
        &region_schema,
        &["order_id", "region"],
    )));
    let first_join = builder.add_join(
        lineitem,
        orders,
        JoinSpec::new(JoinType::Inner, vec![join_order_id_key()])
            .with_output_projection(vec![
                JoinOutputColumn::left(nexus_query_engine::ColumnRef::input(0, "order_id", "order_id"), "order_id"),
                JoinOutputColumn::left(nexus_query_engine::ColumnRef::input(0, "qty", "qty"), "qty"),
                JoinOutputColumn::right(nexus_query_engine::ColumnRef::input(1, "category", "category"), "category"),
            ])
            .with_build_side(JoinSide::Right),
        first_join_schema,
    );
    let second_join = builder.add_join(
        first_join,
        regions,
        JoinSpec::new(JoinType::Inner, vec![join_order_id_key()])
            .with_output_projection(vec![
                JoinOutputColumn::left(nexus_query_engine::ColumnRef::input(0, "order_id", "order_id"), "order_id"),
                JoinOutputColumn::left(nexus_query_engine::ColumnRef::input(0, "qty", "qty"), "qty"),
                JoinOutputColumn::right(nexus_query_engine::ColumnRef::input(1, "region", "region"), "region"),
            ])
            .with_build_side(JoinSide::Right),
        second_join_schema,
    );
    let aggregate = builder.add_aggregate(
        second_join,
        AggregateSpec::new(
            vec![Expr::column("region")],
            vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("qty"), "line_count")],
        ),
        Arc::clone(&aggregate_schema),
    );
    let sort = builder.add_sort(aggregate, sort, Arc::clone(&aggregate_schema));
    let plan = builder.finish_with_host_sink(sort);
    assert!(!plan.capability_analysis().admits_root_join());
    assert!(!plan.capability_analysis().admits_single_join_post_root_pipeline());
    assert!(plan.capability_analysis().admits_closed_multi_join_subgraph());

    let context = bounded_join_execution_context(&runtime);
    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.memory_control_policy.as_str(), "bounded");
    assert_eq!(metrics.parquet_files, 3);
    assert_eq!(metrics.reservation_denied_count, 0);
    assert_eq!(metrics.compiled_pipeline_attempts, 1);
    assert!(metrics.compiled_pipeline_tasks_executed > 0);
    assert_eq!(metrics.compiled_pipeline_host_arrow_sinks, 1);
    assert_eq!(metrics.hash_join_build_count, 2);
    assert_eq!(metrics.operators.iter().filter(|operator| operator.join.is_some()).count(), 2);
    assert!(metrics.operators.iter().filter_map(|operator| operator.join.as_ref()).all(|join| {
        join.execution_strategy == "bounded_reusable_hash_join_single_pass_probe"
            && join.key_count == 1
            && !join.residual_predicate
            && join.gather_map_bytes > 0
    }));
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
            (ReservationKind::GatherMap, "join_gather_map_stream_slice"),
            (ReservationKind::ChunkOperator, "join_output_string_length_sizing"),
            (ReservationKind::JoinOutput, "join_output_stream_chunk"),
            (ReservationKind::SourceRead, "join_source_read"),
            (ReservationKind::BuildSide, "join_build_side"),
            (ReservationKind::HashTable, "join_hash_table"),
            (ReservationKind::ProbeSide, "join_probe_side"),
            (ReservationKind::GatherMap, "join_gather_map"),
            (ReservationKind::ChunkOperator, "join_output_string_length_sizing"),
            (ReservationKind::GatherMap, "join_gather_map_stream_slice"),
            (ReservationKind::ChunkOperator, "join_output_string_length_sizing"),
            (ReservationKind::JoinOutput, "join_output_stream_chunk"),
            (ReservationKind::ChunkOperator, "compiled_aggregate_string_length_sizing"),
            (ReservationKind::PartialState, "compiled_aggregate_partial_state_work"),
            (ReservationKind::PartialState, "compiled_aggregate_partial_state"),
            (ReservationKind::FinalMerge, "compiled_aggregate_final_projection"),
            (ReservationKind::OperatorOutput, "compiled_aggregate_output"),
            (ReservationKind::FinalSort, "final_sort"),
            (ReservationKind::FinalSort, "final_sort"),
            (ReservationKind::Materialization, "host_materialization"),
        ],
    );

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 2);
    assert_eq!(batch.num_columns(), 2);
    assert_eq!(string_array(batch, 0).iter().collect::<Vec<_>>(), vec![Some("east"), Some("west")]);
    assert_eq!(int64_array(batch, 1).values().as_ref(), &[2, 1]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_pure_hash_multi_join_reuses_source_cache_aliases_without_transfer_panic() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let dir = unique_temp_dir("nexus-query-engine-bounded-source-cache-hash-multi-join");
    let left_path = dir.join("lineitem.parquet");
    let orders_path = dir.join("orders.parquet");
    let region_path = dir.join("regions.parquet");
    let left_schema = write_join_left_parquet(&runtime, &left_path);
    let orders_schema = write_join_right_parquet(&runtime, &orders_path);
    let region_schema = write_join_region_parquet(&runtime, &region_path);
    let first_join_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
        Field::new("category", DataType::Utf8, false),
    ]));
    let second_join_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
        Field::new("region", DataType::Utf8, false),
    ]));
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("region", DataType::Utf8, false),
        Field::new("line_count", DataType::Int64, false),
    ]));
    let cache = Arc::new(GpuLocalParquetSourceCache::new(64 * 1024 * 1024));

    let build_plan = || {
        let sort = SortSpec::new(vec![SortExpr::new(Expr::column("region"), false, false)]);
        let mut builder = QueryPlanBuilder::new();
        let lineitem = builder.add_source(SourceNode::LocalParquet(join_left_source(
            &left_path,
            &left_schema,
            &["order_id", "qty"],
        )));
        let orders = builder.add_source(SourceNode::LocalParquet(join_right_source(
            &orders_path,
            &orders_schema,
            &["order_id", "category"],
        )));
        let regions = builder.add_source(SourceNode::LocalParquet(join_right_source(
            &region_path,
            &region_schema,
            &["order_id", "region"],
        )));
        let first_join = builder.add_join(
            lineitem,
            orders,
            JoinSpec::new(JoinType::Inner, vec![join_order_id_key()])
                .with_output_projection(vec![
                    JoinOutputColumn::left(nexus_query_engine::ColumnRef::input(0, "order_id", "order_id"), "order_id"),
                    JoinOutputColumn::left(nexus_query_engine::ColumnRef::input(0, "qty", "qty"), "qty"),
                    JoinOutputColumn::right(
                        nexus_query_engine::ColumnRef::input(1, "category", "category"),
                        "category",
                    ),
                ])
                .with_build_side(JoinSide::Right),
            Arc::clone(&first_join_schema),
        );
        let second_join = builder.add_join(
            first_join,
            regions,
            JoinSpec::new(JoinType::Inner, vec![join_order_id_key()])
                .with_output_projection(vec![
                    JoinOutputColumn::left(nexus_query_engine::ColumnRef::input(0, "order_id", "order_id"), "order_id"),
                    JoinOutputColumn::left(nexus_query_engine::ColumnRef::input(0, "qty", "qty"), "qty"),
                    JoinOutputColumn::right(nexus_query_engine::ColumnRef::input(1, "region", "region"), "region"),
                ])
                .with_build_side(JoinSide::Right),
            Arc::clone(&second_join_schema),
        );
        let aggregate = builder.add_aggregate(
            second_join,
            AggregateSpec::new(
                vec![Expr::column("region")],
                vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("qty"), "line_count")],
            ),
            Arc::clone(&aggregate_schema),
        );
        let sort = builder.add_sort(aggregate, sort, Arc::clone(&aggregate_schema));
        builder.finish_with_host_sink(sort)
    };
    let build_context = || {
        ExecutionContext::builder()
            .with_shared_runtime(Arc::clone(&runtime))
            .with_memory_control_policy(MemoryControlPolicy::Bounded {
                device_budget_bytes: 8 * 1024 * 1024 * 1024,
                max_source_chunk_bytes: 1024,
                max_row_groups_per_chunk: 1,
                max_retry_attempts: 1,
                min_retry_chunk_row_groups: 1,
            })
            .with_local_parquet_source_cache(Arc::clone(&cache))
            .build()
            .unwrap()
    };

    let warm = execute_query_plan(&build_context(), &build_plan()).unwrap();
    assert_eq!(warm.metrics().source_cache_hits, 0);
    assert!(warm.metrics().source_cache_inserts > 0);

    let cached = execute_query_plan(&build_context(), &build_plan()).unwrap();
    let metrics = cached.metrics();
    assert!(metrics.source_cache_hits > 0);
    assert_eq!(metrics.reservation_denied_count, 0);
    assert!(
        metrics.reservation_events.iter().any(|event| {
            event.action == MemoryReservationEventAction::AliasRelabel
                && matches!(event.reservation_kind, ReservationKind::BuildSide | ReservationKind::ProbeSide)
        }),
        "cached join leaf input should relabel the source-cache alias instead of transferring it"
    );

    let QueryOutput::HostArrow(batch) = cached.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 2);
    assert_eq!(string_array(batch, 0).iter().collect::<Vec<_>>(), vec![Some("east"), Some("west")]);
    assert_eq!(int64_array(batch, 1).values().as_ref(), &[2, 1]);

    fs::remove_dir_all(dir).unwrap();
}
