use super::*;

type RollupRankRow = (Option<String>, Option<String>, Option<String>, u32, i64, u64);

#[test]
fn bounded_stddev_join_aggregate_keeps_exact_and_tolerant_statistical_contracts_separate() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let dir = unique_temp_dir("nexus-query-engine-bounded-exact-stddev-join-aggregate");
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
        Field::new("qty_stddev", DataType::Float64, true),
    ]));
    let sort = SortSpec::new(vec![SortExpr::new(Expr::column("qty_stddev"), true, true)]);

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
                JoinOutputColumn::left(
                    nexus_query_engine::expr::ColumnRef::input(0, "order_id", "order_id"),
                    "order_id",
                ),
                JoinOutputColumn::left(nexus_query_engine::expr::ColumnRef::input(0, "qty", "qty"), "qty"),
            ])
            .with_build_side(JoinSide::Right),
        join_schema,
    );
    let aggregate = builder.add_aggregate(
        join,
        AggregateSpec::new(
            vec![Expr::column("order_id")],
            vec![AggregateExpr::new(AggregateFunction::StddevSamp, Expr::column("qty"), "qty_stddev")],
        ),
        Arc::clone(&aggregate_schema),
    );
    let sort = builder.add_sort(aggregate, sort, Arc::clone(&aggregate_schema));
    let plan = builder.finish_with_host_sink(sort);

    let tolerant_context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 1024 * 1024,
            max_source_chunk_bytes: 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .with_statistical_aggregate_execution_mode(StddevSampExecutionMode::GpuNativeTolerant)
        .build()
        .unwrap();
    let tolerant_result = execute_query_plan(&tolerant_context, &plan).unwrap();
    let tolerant_metrics = tolerant_result.metrics();
    assert_eq!(tolerant_metrics.compiled_pipeline_attempts, 1);
    assert!(tolerant_metrics.aggregate_strategy_decisions.iter().any(|decision| {
        decision
            .facts
            .iter()
            .any(|(key, value)| key == "statistical_aggregate_execution_mode" && value == "gpu_native_tolerant")
            && decision
                .facts
                .iter()
                .any(|(key, value)| key == "statistical_aggregate_tolerance_contract" && value == "float_tolerant")
            && decision.facts.iter().any(|(key, value)| key == "stddev_samp_ddof" && value == "1")
    }));
    assert!(tolerant_metrics.reservation_events.iter().all(|event| {
        event.memory_boundary != "host_welford_input_materialization_preflight"
            && event.memory_boundary != "host_welford_output_upload_reservation"
    }));

    let context = bounded_join_execution_context(&runtime);
    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.compiled_pipeline_attempts, 1);
    assert!(metrics.aggregate_strategy_decisions.iter().any(|decision| {
        decision.facts.iter().any(|(key, value)| {
            key == "statistical_aggregate_execution_mode" && value == "datafusion_exact_host_welford"
        })
    }));
    assert!(metrics.reservation_events.iter().any(|event| {
        event.action == MemoryReservationEventAction::Reserve
            && event.outcome.as_str() == "granted"
            && event.reservation_kind == ReservationKind::Materialization
            && event.memory_boundary == "host_welford_input_materialization_preflight"
    }));
    assert!(metrics.reservation_events.iter().any(|event| {
        event.action == MemoryReservationEventAction::Reserve
            && event.outcome.as_str() == "granted"
            && event.reservation_kind == ReservationKind::OperatorOutput
            && event.memory_boundary == "host_welford_output_upload_reservation"
    }));
    assert!(metrics.reservation_events.iter().any(|event| {
        event.action == MemoryReservationEventAction::Transfer
            && event.outcome.as_str() == "applied"
            && event.reservation_kind == ReservationKind::OperatorOutput
            && event.memory_boundary == "host_welford_output_upload_reservation"
    }));

    fs::remove_dir_all(dir).unwrap();
}
