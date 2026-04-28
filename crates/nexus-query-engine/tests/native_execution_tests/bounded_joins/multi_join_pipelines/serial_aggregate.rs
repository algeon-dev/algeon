use super::*;

#[test]
fn bounded_native_query_executes_serial_aggregate_multi_join_distinct_count_spine() {
    let runtime = backend_runtime();
    let dir = unique_temp_dir("nexus-query-engine-bounded-serial-aggregate-multi-join-spine");
    let part_path = dir.join("part.parquet");
    let partsupp_path = dir.join("partsupp.parquet");
    let supplier_path = dir.join("supplier.parquet");
    let part_schema = write_part_distinct_count_part_parquet(&runtime, &part_path);
    let partsupp_schema = write_part_distinct_count_partsupp_parquet(&runtime, &partsupp_path);
    let supplier_schema = write_part_distinct_count_supplier_parquet(&runtime, &supplier_path);
    let join_schema = Arc::new(Schema::new(vec![
        Field::new("ps_suppkey", DataType::Int32, false),
        Field::new("p_brand", DataType::Utf8, false),
        Field::new("p_type", DataType::Utf8, false),
        Field::new("p_size", DataType::Int32, false),
    ]));
    let distinct_schema = Arc::new(Schema::new(vec![
        Field::new("p_brand", DataType::Utf8, false),
        Field::new("p_type", DataType::Utf8, false),
        Field::new("p_size", DataType::Int32, false),
        Field::new("ps_suppkey", DataType::Int32, false),
    ]));
    let counted_schema = Arc::new(Schema::new(vec![
        Field::new("p_brand", DataType::Utf8, false),
        Field::new("p_type", DataType::Utf8, false),
        Field::new("p_size", DataType::Int32, false),
        Field::new("supplier_cnt", DataType::Int64, false),
    ]));
    let projected_schema = Arc::new(Schema::new(vec![
        Field::new("supplier_cnt", DataType::Int64, false),
        Field::new("p_brand", DataType::Utf8, false),
        Field::new("p_type", DataType::Utf8, false),
        Field::new("p_size", DataType::Int32, false),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let partsupp = builder.add_source(SourceNode::LocalParquet(join_left_source(
        &partsupp_path,
        &partsupp_schema,
        &["ps_partkey", "ps_suppkey"],
    )));
    let part = builder.add_source(SourceNode::LocalParquet(join_right_source(
        &part_path,
        &part_schema,
        &["p_partkey", "p_brand", "p_type", "p_size"],
    )));
    let supplier = builder.add_source(SourceNode::LocalParquet(join_right_source(
        &supplier_path,
        &supplier_schema,
        &["s_suppkey"],
    )));
    let part_join = builder.add_join(
        partsupp,
        part,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                nexus_query_engine::expr::ColumnRef::input(0, "ps_partkey", "ps_partkey"),
                nexus_query_engine::expr::ColumnRef::input(1, "p_partkey", "p_partkey"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(
                nexus_query_engine::expr::ColumnRef::input(0, "ps_suppkey", "ps_suppkey"),
                "ps_suppkey",
            ),
            JoinOutputColumn::right(nexus_query_engine::expr::ColumnRef::input(1, "p_brand", "p_brand"), "p_brand"),
            JoinOutputColumn::right(nexus_query_engine::expr::ColumnRef::input(1, "p_type", "p_type"), "p_type"),
            JoinOutputColumn::right(nexus_query_engine::expr::ColumnRef::input(1, "p_size", "p_size"), "p_size"),
        ])
        .with_build_side(JoinSide::Right),
        Arc::clone(&join_schema),
    );
    let supplier_join = builder.add_join(
        part_join,
        supplier,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                nexus_query_engine::expr::ColumnRef::input(0, "ps_suppkey", "ps_suppkey"),
                nexus_query_engine::expr::ColumnRef::input(1, "s_suppkey", "s_suppkey"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(
                nexus_query_engine::expr::ColumnRef::input(0, "ps_suppkey", "ps_suppkey"),
                "ps_suppkey",
            ),
            JoinOutputColumn::left(nexus_query_engine::expr::ColumnRef::input(0, "p_brand", "p_brand"), "p_brand"),
            JoinOutputColumn::left(nexus_query_engine::expr::ColumnRef::input(0, "p_type", "p_type"), "p_type"),
            JoinOutputColumn::left(nexus_query_engine::expr::ColumnRef::input(0, "p_size", "p_size"), "p_size"),
        ])
        .with_build_side(JoinSide::Right),
        Arc::clone(&join_schema),
    );
    let distinct_suppliers = builder.add_aggregate(
        supplier_join,
        AggregateSpec::new(
            vec![Expr::column("p_brand"), Expr::column("p_type"), Expr::column("p_size"), Expr::column("ps_suppkey")],
            vec![],
        ),
        distinct_schema,
    );
    let supplier_counts = builder.add_aggregate(
        distinct_suppliers,
        AggregateSpec::new(
            vec![Expr::column("p_brand"), Expr::column("p_type"), Expr::column("p_size")],
            vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("ps_suppkey"), "supplier_cnt")],
        ),
        Arc::clone(&counted_schema),
    );
    let projection = builder.add_projection(
        supplier_counts,
        projection_outputs(
            vec![Expr::column("supplier_cnt"), Expr::column("p_brand"), Expr::column("p_type"), Expr::column("p_size")],
            Arc::clone(&projected_schema),
        ),
    );
    let sort = builder.add_sort(
        projection,
        SortSpec::new(vec![
            SortExpr::new(Expr::column("supplier_cnt"), true, false),
            SortExpr::new(Expr::column("p_brand"), false, false),
            SortExpr::new(Expr::column("p_type"), false, false),
            SortExpr::new(Expr::column("p_size"), false, false),
        ]),
        projected_schema,
    );
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
    assert_eq!(metrics.hash_join_build_count, 2);
    assert_eq!(metrics.compiled_pipeline_attempts, 1);
    assert!(metrics.compiled_pipeline_tasks_executed > 0);
    assert_eq!(metrics.compiled_pipeline_host_arrow_sinks, 1);
    assert_eq!(metrics.operators.iter().filter(|operator| operator.join.is_some()).count(), 2);
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
            (ReservationKind::ChunkOperator, "join_output_string_length_sizing"),
            (ReservationKind::ChunkOperator, "join_output_string_length_sizing"),
            (ReservationKind::SourceRead, "join_source_read"),
            (ReservationKind::BuildSide, "join_build_side"),
            (ReservationKind::HashTable, "join_hash_table"),
            (ReservationKind::GatherMap, "join_gather_map_stream_slice"),
            (ReservationKind::ChunkOperator, "join_output_string_length_sizing"),
            (ReservationKind::ChunkOperator, "join_output_string_length_sizing"),
            (ReservationKind::JoinOutput, "join_output_stream_chunk"),
            (ReservationKind::ProbeSide, "join_probe_side"),
            (ReservationKind::GatherMap, "join_gather_map"),
            (ReservationKind::ChunkOperator, "join_output_string_length_sizing"),
            (ReservationKind::ChunkOperator, "join_output_string_length_sizing"),
            (ReservationKind::GatherMap, "join_gather_map_stream_slice"),
            (ReservationKind::ChunkOperator, "join_output_string_length_sizing"),
            (ReservationKind::ChunkOperator, "join_output_string_length_sizing"),
            (ReservationKind::JoinOutput, "join_output_stream_chunk"),
            (ReservationKind::ChunkOperator, "compiled_aggregate_string_length_sizing"),
            (ReservationKind::ChunkOperator, "compiled_aggregate_string_length_sizing"),
            (ReservationKind::PartialState, "compiled_aggregate_partial_state_work"),
            (ReservationKind::PartialState, "compiled_aggregate_partial_state"),
            (ReservationKind::FinalMerge, "compiled_aggregate_final_projection"),
            (ReservationKind::OperatorOutput, "compiled_aggregate_output"),
            (ReservationKind::OperatorOutput, "compiled_aggregate"),
            (ReservationKind::OperatorOutput, "compiled_aggregate_output"),
            (ReservationKind::OperatorOutput, "compiled_projection"),
            (ReservationKind::OperatorOutput, "compiled_projection_output"),
            (ReservationKind::FinalSort, "final_sort"),
            (ReservationKind::FinalSort, "final_sort"),
            (ReservationKind::Materialization, "host_materialization"),
        ],
    );

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 2);
    assert_eq!(batch.num_columns(), 4);
    assert_eq!(int64_array(batch, 0).values().as_ref(), &[3, 1]);
    assert_eq!(string_array(batch, 1).iter().collect::<Vec<_>>(), vec![Some("Brand#11"), Some("Brand#22")]);
    assert_eq!(string_array(batch, 2).iter().collect::<Vec<_>>(), vec![Some("SMALL BRUSHED"), Some("MEDIUM POLISHED")]);
    assert_eq!(int32_array(batch, 3).values().as_ref(), &[1, 2]);

    fs::remove_dir_all(dir).unwrap();
}
