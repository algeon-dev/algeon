use super::*;

#[test]
fn bounded_native_query_preserves_right_then_left_projected_cross_output_order() {
    let runtime = backend_runtime();
    let dir = unique_temp_dir("nexus-query-engine-bounded-root-projected-cross-join");
    let left_path = dir.join("lineitem.parquet");
    let right_path = dir.join("orders.parquet");
    let left_schema = write_join_left_parquet(&runtime, &left_path);
    let right_schema = write_join_right_parquet(&runtime, &right_path);
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("category", DataType::Utf8, false),
        Field::new("qty", DataType::Int32, false),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(join_left_source(&left_path, &left_schema, &["qty"])));
    let right =
        builder.add_source(SourceNode::LocalParquet(join_right_source(&right_path, &right_schema, &["category"])));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(JoinType::Cross, vec![]).with_output_projection(vec![
            JoinOutputColumn::right(nexus_query_engine::expr::ColumnRef::input(1, "category", "category"), "category"),
            JoinOutputColumn::left(nexus_query_engine::expr::ColumnRef::input(0, "qty", "qty"), "qty"),
        ]),
        Arc::clone(&output_schema),
    );
    let plan = builder.finish_with_host_sink(join);
    assert!(plan.capability_analysis().admits_root_join());

    let context = bounded_join_execution_context(&runtime);
    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.hash_join_build_count, 0);
    assert_eq!(metrics.filtered_join_build_count, 0);
    assert_eq!(metrics.mixed_join_size_data_bytes, 0);
    assert!(metrics.reservation_events.iter().all(|event| event.reservation_kind != ReservationKind::GatherMap));
    assert_reservation_kind_subsequence(
        metrics,
        &[
            ReservationKind::BuildSide,
            ReservationKind::ProbeSide,
            ReservationKind::JoinOutput,
            ReservationKind::Materialization,
        ],
    );
    let join_metrics =
        metrics.operators.iter().find_map(|operator| operator.join.as_ref()).expect("join metrics should be recorded");
    assert_eq!(join_metrics.join_type, "cross");
    assert_eq!(join_metrics.execution_strategy, "bounded_projected_cross_join");
    assert_eq!(join_metrics.key_count, 0);
    assert!(!join_metrics.residual_predicate);
    assert_eq!(join_metrics.hash_table_bytes, 0);
    assert_eq!(join_metrics.filtered_probe_state_bytes, 0);
    assert_eq!(join_metrics.mixed_size_data_bytes, 0);
    assert_eq!(join_metrics.gather_map_bytes, 0);
    assert_eq!(join_metrics.output_rows, 12);
    assert_eq!(
        granted_reservation_requested_bytes(metrics, ReservationKind::JoinOutput, "join_output"),
        join_metrics.output_bytes
    );

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 12);
    assert_eq!(batch.num_columns(), 2);
    assert_eq!(batch.schema().field(0).name(), "category");
    assert_eq!(batch.schema().field(1).name(), "qty");
    let mut rows = (0..batch.num_rows())
        .map(|row| (string_array(batch, 0).value(row).to_string(), int32_array(batch, 1).value(row)))
        .collect::<Vec<_>>();
    rows.sort();
    assert_eq!(
        rows,
        vec![
            ("four".to_string(), 10),
            ("four".to_string(), 20),
            ("four".to_string(), 25),
            ("four".to_string(), 30),
            ("three".to_string(), 10),
            ("three".to_string(), 20),
            ("three".to_string(), 25),
            ("three".to_string(), 30),
            ("two".to_string(), 10),
            ("two".to_string(), 20),
            ("two".to_string(), 25),
            ("two".to_string(), 30),
        ]
    );

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_preserves_mixed_projected_cross_output_order() {
    let runtime = backend_runtime();
    let dir = unique_temp_dir("nexus-query-engine-bounded-root-projected-cross-join-mixed-order");
    let left_path = dir.join("lineitem.parquet");
    let right_path = dir.join("orders.parquet");
    let left_schema = write_join_left_parquet(&runtime, &left_path);
    let right_schema = write_join_right_parquet(&runtime, &right_path);
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("right_order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
        Field::new("category", DataType::Utf8, false),
        Field::new("left_order_id", DataType::Int32, false),
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
        JoinSpec::new(JoinType::Cross, vec![]).with_output_projection(vec![
            JoinOutputColumn::right(
                nexus_query_engine::expr::ColumnRef::input(1, "order_id", "right_order_id"),
                "right_order_id",
            ),
            JoinOutputColumn::left(nexus_query_engine::expr::ColumnRef::input(0, "qty", "qty"), "qty"),
            JoinOutputColumn::right(nexus_query_engine::expr::ColumnRef::input(1, "category", "category"), "category"),
            JoinOutputColumn::left(
                nexus_query_engine::expr::ColumnRef::input(0, "order_id", "left_order_id"),
                "left_order_id",
            ),
        ]),
        Arc::clone(&output_schema),
    );
    let plan = builder.finish_with_host_sink(join);
    assert!(plan.capability_analysis().admits_root_join());

    let context = bounded_join_execution_context(&runtime);
    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    let join_metrics =
        metrics.operators.iter().find_map(|operator| operator.join.as_ref()).expect("join metrics should be recorded");
    assert_eq!(join_metrics.join_type, "cross");
    assert_eq!(join_metrics.execution_strategy, "bounded_projected_cross_join");
    assert_eq!(join_metrics.output_rows, 12);
    assert_eq!(join_metrics.gather_map_bytes, 0);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.schema().field(0).name(), "right_order_id");
    assert_eq!(batch.schema().field(1).name(), "qty");
    assert_eq!(batch.schema().field(2).name(), "category");
    assert_eq!(batch.schema().field(3).name(), "left_order_id");
    let mut rows = (0..batch.num_rows())
        .map(|row| {
            (
                int32_array(batch, 0).value(row),
                int32_array(batch, 1).value(row),
                string_array(batch, 2).value(row).to_string(),
                int32_array(batch, 3).value(row),
            )
        })
        .collect::<Vec<_>>();
    rows.sort();
    assert_eq!(
        rows,
        vec![
            (2, 10, "two".to_string(), 1),
            (2, 20, "two".to_string(), 2),
            (2, 25, "two".to_string(), 2),
            (2, 30, "two".to_string(), 3),
            (3, 10, "three".to_string(), 1),
            (3, 20, "three".to_string(), 2),
            (3, 25, "three".to_string(), 2),
            (3, 30, "three".to_string(), 3),
            (4, 10, "four".to_string(), 1),
            (4, 20, "four".to_string(), 2),
            (4, 25, "four".to_string(), 2),
            (4, 30, "four".to_string(), 3),
        ]
    );

    fs::remove_dir_all(dir).unwrap();
}
