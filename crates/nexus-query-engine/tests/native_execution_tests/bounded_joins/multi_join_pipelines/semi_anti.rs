use super::*;

#[test]
fn bounded_native_query_executes_mixed_left_semi_left_anti_multi_join_pipeline_with_ordered_reservations() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let dir = unique_temp_dir("nexus-query-engine-bounded-mixed-semi-anti-multi-join");
    let left_path = dir.join("lineitem.parquet");
    let semi_path = dir.join("semi_orders.parquet");
    let anti_path = dir.join("anti_orders.parquet");
    let left_schema = write_join_left_parquet(&runtime, &left_path);
    let semi_schema = write_join_right_parquet(&runtime, &semi_path);
    let anti_schema = write_join_right_keys_parquet(&runtime, &anti_path, &[3, 5]);
    let left_only_join_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let lineitem =
        builder.add_source(SourceNode::LocalParquet(join_left_source(&left_path, &left_schema, &["order_id", "qty"])));
    let semi_orders =
        builder.add_source(SourceNode::LocalParquet(join_right_source(&semi_path, &semi_schema, &["order_id"])));
    let anti_orders =
        builder.add_source(SourceNode::LocalParquet(join_right_source(&anti_path, &anti_schema, &["order_id"])));
    let semi_join = builder.add_join(
        lineitem,
        semi_orders,
        JoinSpec::new(JoinType::LeftSemi, vec![join_order_id_key()])
            .with_residual_predicate(qty_gt_right_order_id_times_ten())
            .with_output_projection(vec![
                JoinOutputColumn::left(nexus_query_engine::ColumnRef::input(0, "order_id", "order_id"), "order_id"),
                JoinOutputColumn::left(nexus_query_engine::ColumnRef::input(0, "qty", "qty"), "qty"),
            ])
            .with_build_side(JoinSide::Right),
        Arc::clone(&left_only_join_schema),
    );
    let anti_join = builder.add_join(
        semi_join,
        anti_orders,
        JoinSpec::new(JoinType::LeftAnti, vec![join_order_id_key()])
            .with_residual_predicate(qty_gt_right_order_id_times_ten())
            .with_output_projection(vec![
                JoinOutputColumn::left(nexus_query_engine::ColumnRef::input(0, "order_id", "order_id"), "order_id"),
                JoinOutputColumn::left(nexus_query_engine::ColumnRef::input(0, "qty", "qty"), "qty"),
            ])
            .with_build_side(JoinSide::Right),
        Arc::clone(&left_only_join_schema),
    );
    let plan = builder.finish_with_host_sink(anti_join);
    assert!(!plan.capability_analysis().admits_root_join());
    assert!(!plan.capability_analysis().admits_single_join_post_root_pipeline());
    assert!(plan.capability_analysis().admits_closed_multi_join_subgraph());
    assert!(plan.bounded_join_unsupported_shape_facts(ExecutionBoundaryPolicy::default()).is_none());

    let context = bounded_join_execution_context(&runtime);
    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.memory_control_policy.as_str(), "bounded");
    assert_eq!(metrics.parquet_files, 3);
    assert_eq!(metrics.reservation_denied_count, 0);
    assert_eq!(metrics.hash_join_build_count, 0);
    assert_eq!(metrics.hash_join_hash_table_bytes, 0);
    assert!(metrics.mixed_join_size_data_bytes > 0);
    assert_eq!(metrics.operators.iter().filter(|operator| operator.join.is_some()).count(), 2);
    let join_metrics = metrics.operators.iter().filter_map(|operator| operator.join.as_ref()).collect::<Vec<_>>();
    assert_eq!(join_metrics[0].execution_strategy, "bounded_mixed_keyed_residual_join");
    assert_eq!(join_metrics[0].join_type, "left_semi");
    assert_eq!(join_metrics[0].build_side, "right");
    assert_eq!(join_metrics[0].probe_side, "left");
    assert_eq!(join_metrics[0].key_count, 1);
    assert!(join_metrics[0].residual_predicate);
    assert_eq!(join_metrics[0].hash_table_bytes, 0);
    assert_eq!(join_metrics[0].output_rows, 1);
    assert!(join_metrics[0].mixed_size_data_bytes > 0);
    assert!(join_metrics[0].gather_map_bytes > 0);
    assert_eq!(join_metrics[1].execution_strategy, "bounded_mixed_keyed_residual_join");
    assert_eq!(join_metrics[1].join_type, "left_anti");
    assert_eq!(join_metrics[1].build_side, "right");
    assert_eq!(join_metrics[1].probe_side, "left");
    assert_eq!(join_metrics[1].key_count, 1);
    assert!(join_metrics[1].residual_predicate);
    assert_eq!(join_metrics[1].hash_table_bytes, 0);
    assert_eq!(join_metrics[1].output_rows, 1);
    assert!(join_metrics[1].mixed_size_data_bytes > 0);
    assert!(join_metrics[1].gather_map_bytes > 0);
    assert_eq!(
        join_metrics[0].mixed_size_data_bytes + join_metrics[1].mixed_size_data_bytes,
        metrics.mixed_join_size_data_bytes
    );
    assert_granted_reservation_trace(
        metrics,
        &[
            (ReservationKind::SourceRead, "join_source_read"),
            (ReservationKind::SourceRead, "join_source_read"),
            (ReservationKind::BuildSide, "join_build_side"),
            (ReservationKind::ProbeSide, "join_probe_side"),
            (ReservationKind::MixedSizeData, "join_mixed_size_data"),
            (ReservationKind::GatherMap, "join_gather_map"),
            (ReservationKind::JoinOutput, "join_output"),
            (ReservationKind::SourceRead, "join_source_read"),
            (ReservationKind::BuildSide, "join_build_side"),
            (ReservationKind::ProbeSide, "join_probe_side"),
            (ReservationKind::MixedSizeData, "join_mixed_size_data"),
            (ReservationKind::GatherMap, "join_gather_map"),
            (ReservationKind::JoinOutput, "join_output"),
            (ReservationKind::Materialization, "host_materialization"),
        ],
    );
    let mixed_size_data = granted_reservation_requested_bytes_for_boundary(
        metrics,
        ReservationKind::MixedSizeData,
        "join_mixed_size_data",
    );
    assert_eq!(mixed_size_data.len(), 2);
    assert!(mixed_size_data.iter().all(|bytes| *bytes > 0));
    let gather_maps =
        granted_reservation_requested_bytes_for_boundary(metrics, ReservationKind::GatherMap, "join_gather_map");
    assert_eq!(gather_maps, vec![cudf::DataFrameJoinIndices::reservation_size_bytes_for_len(1).unwrap() as u64; 2]);
    let join_outputs =
        granted_reservation_requested_bytes_for_boundary(metrics, ReservationKind::JoinOutput, "join_output");
    assert_eq!(join_outputs.len(), 2);
    assert!(join_outputs.iter().all(|bytes| *bytes > 0));

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 1);
    assert_eq!(batch.num_columns(), 2);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[2]);
    assert_eq!(int32_array(batch, 1).values().as_ref(), &[25]);

    fs::remove_dir_all(dir).unwrap();
}
