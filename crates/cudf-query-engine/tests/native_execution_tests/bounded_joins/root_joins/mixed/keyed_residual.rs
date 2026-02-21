use super::*;
#[test]
fn bounded_native_query_executes_root_mixed_inner_join_with_ordered_reservations() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let dir = unique_temp_dir("cudf-query-engine-bounded-root-mixed-inner-join");
    let left_path = dir.join("lineitem.parquet");
    let right_path = dir.join("orders.parquet");
    let left_schema = write_join_left_parquet(&runtime, &left_path);
    let right_schema = write_join_right_parquet(&runtime, &right_path);
    let output_schema = Arc::new(Schema::new(vec![
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
            .with_residual_predicate(qty_gt_right_order_id_times_ten())
            .with_output_projection(vec![
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
    assert_eq!(metrics.reservation_denied_count, 0);
    assert_eq!(metrics.hash_join_build_count, 0);
    assert!(metrics.mixed_join_size_data_bytes > 0);
    assert_reservation_kind_subsequence(
        metrics,
        &[
            ReservationKind::BuildSide,
            ReservationKind::ProbeSide,
            ReservationKind::MixedSizeData,
            ReservationKind::GatherMap,
            ReservationKind::JoinOutput,
            ReservationKind::Materialization,
        ],
    );
    let join_metrics =
        metrics.operators.iter().find_map(|operator| operator.join.as_ref()).expect("join metrics should be recorded");
    assert_eq!(join_metrics.join_type, "inner");
    assert_eq!(join_metrics.execution_strategy, "bounded_mixed_keyed_residual_join");
    assert_eq!(join_metrics.key_count, 1);
    assert!(join_metrics.residual_predicate);
    assert_eq!(join_metrics.hash_table_bytes, 0);
    assert_eq!(join_metrics.mixed_size_data_bytes, metrics.mixed_join_size_data_bytes);
    assert!(join_metrics.gather_map_bytes > 0);
    assert_eq!(join_metrics.output_rows, 1);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 1);
    assert_eq!(int32_array(batch, 0).value(0), 25);
    assert_eq!(string_array(batch, 1).value(0), "two");

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_executes_root_mixed_left_join_with_unmatched_left_rows() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let dir = unique_temp_dir("cudf-query-engine-bounded-root-mixed-left-join");
    let left_path = dir.join("lineitem.parquet");
    let right_path = dir.join("orders.parquet");
    let left_schema = write_join_left_parquet(&runtime, &left_path);
    let right_schema = write_join_right_parquet(&runtime, &right_path);
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("qty", DataType::Int32, false),
        Field::new("category", DataType::Utf8, true),
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
        JoinSpec::new(JoinType::Left, vec![join_order_id_key()])
            .with_residual_predicate(qty_gt_right_order_id_times_ten())
            .with_output_projection(vec![
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
    assert_eq!(metrics.hash_join_hash_table_bytes, 0);
    assert!(metrics.mixed_join_size_data_bytes > 0);
    let join_metrics =
        metrics.operators.iter().find_map(|operator| operator.join.as_ref()).expect("join metrics should be recorded");
    assert_eq!(join_metrics.join_type, "left");
    assert_eq!(join_metrics.execution_strategy, "bounded_mixed_keyed_residual_join");
    assert_eq!(join_metrics.output_rows, 4);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    let category = string_array(batch, 1);
    let mut rows = (0..batch.num_rows())
        .map(|row| {
            let category = if category.is_null(row) { None } else { Some(category.value(row).to_string()) };
            (int32_array(batch, 0).value(row), category)
        })
        .collect::<Vec<_>>();
    rows.sort();
    assert_eq!(rows, vec![(10, None), (20, None), (25, Some("two".to_string())), (30, None),]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_executes_root_mixed_full_join_with_pair_reservations() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let dir = unique_temp_dir("cudf-query-engine-bounded-root-mixed-full-join");
    let left_path = dir.join("lineitem.parquet");
    let right_path = dir.join("orders.parquet");
    let left_schema = write_join_left_parquet(&runtime, &left_path);
    let right_schema = write_join_right_parquet(&runtime, &right_path);
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("left_order_id", DataType::Int32, true),
        Field::new("right_order_id", DataType::Int32, true),
        Field::new("qty", DataType::Int32, true),
        Field::new("category", DataType::Utf8, true),
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
        JoinSpec::new(JoinType::Full, vec![join_order_id_key()])
            .with_residual_predicate(qty_gt_right_order_id_times_ten())
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
    let plan = builder.finish_with_host_sink(join);
    assert!(plan.capability_analysis().admits_root_join());

    let context = bounded_join_execution_context(&runtime);
    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.hash_join_build_count, 0);
    assert!(metrics.mixed_join_size_data_bytes > 0);
    assert_granted_reservation_trace(
        metrics,
        &[
            (ReservationKind::SourceRead, "join_source_read"),
            (ReservationKind::SourceRead, "join_source_read"),
            (ReservationKind::BuildSide, "join_build_side"),
            (ReservationKind::ProbeSide, "join_probe_side"),
            (ReservationKind::MixedSizeData, "join_mixed_size_data"),
            (ReservationKind::GatherMap, "join_gather_map"),
            (ReservationKind::ChunkOperator, "join_output_string_length_sizing"),
            (ReservationKind::JoinOutput, "join_output"),
            (ReservationKind::Materialization, "host_materialization"),
        ],
    );
    let join_metrics =
        metrics.operators.iter().find_map(|operator| operator.join.as_ref()).expect("join metrics should be recorded");
    assert_eq!(join_metrics.join_type, "full");
    assert_eq!(join_metrics.execution_strategy, "bounded_mixed_keyed_residual_join");
    assert_eq!(join_metrics.output_rows, 6);
    assert_eq!(join_metrics.mixed_size_data_bytes, metrics.mixed_join_size_data_bytes);
    assert_eq!(
        granted_reservation_requested_bytes(metrics, ReservationKind::GatherMap, "join_gather_map"),
        cudf::DataFrameJoinPairIndices::reservation_size_bytes_for_len(6).unwrap() as u64
    );

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    let left_order_id = int32_array(batch, 0);
    let right_order_id = int32_array(batch, 1);
    let qty = int32_array(batch, 2);
    let category = string_array(batch, 3);
    let mut rows = (0..batch.num_rows())
        .map(|row| {
            (
                if left_order_id.is_null(row) { None } else { Some(left_order_id.value(row)) },
                if right_order_id.is_null(row) { None } else { Some(right_order_id.value(row)) },
                if qty.is_null(row) { None } else { Some(qty.value(row)) },
                if category.is_null(row) { None } else { Some(category.value(row).to_string()) },
            )
        })
        .collect::<Vec<_>>();
    rows.sort();
    assert_eq!(
        rows,
        vec![
            (None, Some(3), None, Some("three".to_string())),
            (None, Some(4), None, Some("four".to_string())),
            (Some(1), None, Some(10), None),
            (Some(2), None, Some(20), None),
            (Some(2), Some(2), Some(25), Some("two".to_string())),
            (Some(3), None, Some(30), None),
        ]
    );

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_executes_root_mixed_left_semi_and_anti_joins_with_single_gather_map() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let dir = unique_temp_dir("cudf-query-engine-bounded-root-mixed-semi-anti-join");
    let left_path = dir.join("lineitem.parquet");
    let right_path = dir.join("orders.parquet");
    let left_schema = write_join_left_parquet(&runtime, &left_path);
    let right_schema = write_join_right_parquet(&runtime, &right_path);

    let semi = execute_bounded_left_only_residual_join(
        &runtime,
        &left_path,
        &right_path,
        &left_schema,
        &right_schema,
        JoinType::LeftSemi,
    );
    let anti = execute_bounded_left_only_residual_join(
        &runtime,
        &left_path,
        &right_path,
        &left_schema,
        &right_schema,
        JoinType::LeftAnti,
    );

    let QueryOutput::HostArrow(semi_batch) = semi.output() else {
        panic!("left semi bounded mixed join should return host output");
    };
    assert_eq!(int32_array(semi_batch, 0).values().as_ref(), &[25]);
    assert_bounded_mixed_left_only_join_metrics(semi.metrics(), "left_semi", 1);
    assert_eq!(
        granted_reservation_requested_bytes(semi.metrics(), ReservationKind::GatherMap, "join_gather_map"),
        cudf::DataFrameJoinIndices::reservation_size_bytes_for_len(1).unwrap() as u64
    );

    let QueryOutput::HostArrow(anti_batch) = anti.output() else {
        panic!("left anti bounded mixed join should return host output");
    };
    let mut anti_rows = int32_array(anti_batch, 0).values().iter().copied().collect::<Vec<_>>();
    anti_rows.sort();
    assert_eq!(anti_rows, vec![10, 20, 30]);
    assert_bounded_mixed_left_only_join_metrics(anti.metrics(), "left_anti", 3);
    assert_eq!(
        granted_reservation_requested_bytes(anti.metrics(), ReservationKind::GatherMap, "join_gather_map"),
        cudf::DataFrameJoinIndices::reservation_size_bytes_for_len(3).unwrap() as u64
    );

    fs::remove_dir_all(dir).unwrap();
}
