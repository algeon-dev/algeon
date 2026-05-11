#[test]
fn bounded_native_query_splits_join_leaf_source_read_after_reservation_denial() {
    let dir = unique_temp_dir("nexus-query-engine-bounded-root-hash-join-source-retry-split");
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
                JoinOutputColumn::left(
                    nexus_query_engine::expr::ColumnRef::input(0, "order_id", "order_id"),
                    "order_id",
                ),
                JoinOutputColumn::left(nexus_query_engine::expr::ColumnRef::input(0, "qty", "qty"), "qty"),
                JoinOutputColumn::right(
                    nexus_query_engine::expr::ColumnRef::input(1, "category", "category"),
                    "category",
                ),
            ])
            .with_build_side(JoinSide::Right),
        Arc::clone(&output_schema),
    );
    let plan = builder.finish_with_host_sink(join);
    let context = backend_context_with_memory_control_policy(MemoryControlPolicy::Bounded {
        device_budget_bytes: 2_500_000,
        max_source_chunk_bytes: 16 * 1024 * 1024,
        max_row_groups_per_chunk: 4,
        max_retry_attempts: 4,
        min_retry_chunk_row_groups: 1,
    });

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
#[test]
fn bounded_native_query_reports_join_leaf_source_read_hard_budget_failure() {
    let dir = unique_temp_dir("nexus-query-engine-bounded-join-source-hard-budget-failure");
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
                JoinOutputColumn::left(
                    nexus_query_engine::expr::ColumnRef::input(0, "order_id", "order_id"),
                    "order_id",
                ),
                JoinOutputColumn::left(nexus_query_engine::expr::ColumnRef::input(0, "qty", "qty"), "qty"),
                JoinOutputColumn::right(
                    nexus_query_engine::expr::ColumnRef::input(1, "category", "category"),
                    "category",
                ),
            ])
            .with_build_side(JoinSide::Right),
        output_schema,
    );
    let plan = builder.finish_with_host_sink(join);
    let context = backend_context_with_grant(admitted_budget);

    let error = match execute_query_plan(&context, &plan) {
        Ok(_) => panic!("bounded join source read budget should fail"),
        Err(error) => error,
    };
    let admitted_budget =
        u64::try_from(cudf::DataFrame::minimum_parquet_source_read_budget_bytes()).unwrap().to_string();
    assert_eq!(error.code(), QueryErrorCode::MemoryReservationFailed);
    assert_eq!(error.fact_value("budget_bytes"), Some(admitted_budget.as_str()));
    assert_eq!(error.fact_value("memory_boundary"), Some("join_source_read"));
    assert_eq!(error.fact_value("reservation_kind"), Some("source_read"));
    assert_eq!(error.fact_value("stage"), Some("join_source_read"));
    assert_eq!(error.fact_value("chunk_index"), Some("0"));
    assert_eq!(error.fact_value("file_index"), Some("0"));
    assert_eq!(error.fact_value("row_group_indexes"), Some("0"));
    assert_eq!(error.fact_value("row_group_count"), Some("1"));

    fs::remove_dir_all(dir).unwrap();
}
