use super::*;

#[test]
fn bounded_native_query_retries_final_sort_with_hierarchical_topk() {
    let dir = unique_temp_dir("cudf-query-engine-bounded-final-sort-retry");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_single_row_group_final_sort_input_parquet(&path, 4096);
    let aggregated_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("sum_m00", DataType::Int64, true),
    ]));
    let final_sort_schema = wide_final_sort_schema(16);

    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["key".to_string(), "m00".to_string()],
    );
    let aggregate = AggregateSpec::new(
        vec![Expr::column("key")],
        vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("m00"), "sum_m00")],
    );
    let sort = SortSpec::new(vec![SortExpr::new(Expr::column("key"), true, false)]);
    let limit = LimitSpec::new(3, 7);

    let mut builder = QueryPlanBuilder::new();
    let source_id = builder.add_source(SourceNode::LocalParquet(source.clone()));
    let one_chunk_plan =
        plan_local_parquet_source_chunks(source_id, &source, SourceChunkPlanningPolicy::new(None, 1)).unwrap();
    assert_eq!(one_chunk_plan.chunk_count(), 1);
    let aggregate = builder.add_aggregate(source_id, aggregate, Arc::clone(&aggregated_schema));
    let projection = builder
        .add_projection(aggregate, projection_outputs(wide_final_sort_projection(16), Arc::clone(&final_sort_schema)));
    let sort = builder.add_sort(projection, sort, Arc::clone(&final_sort_schema));
    let limit = builder.add_limit(sort, limit, Arc::clone(&final_sort_schema));
    let plan = builder.finish_with_host_sink(limit);

    let baseline_context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 8 * 1024 * 1024 * 1024,
            max_source_chunk_bytes: 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .build()
        .unwrap();
    let baseline = execute_query_plan(&baseline_context, &plan).unwrap();
    let QueryOutput::HostArrow(expected) = baseline.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    let baseline_metrics = baseline.metrics();
    let first_final_sort_event = baseline_metrics
        .reservation_events
        .iter()
        .position(|event| {
            event.outcome.as_str() == "granted"
                && event.reservation_kind == ReservationKind::FinalSort
                && event.memory_boundary == "final_sort"
        })
        .expect("baseline should reserve final sort work");
    let input_bytes = baseline_metrics.reservation_events[..first_final_sort_event]
        .iter()
        .rfind(|event| event.outcome.as_str() == "granted")
        .expect("baseline should have a resident aggregate input before final sort")
        .requested_bytes;
    let final_sort_bytes = baseline_metrics
        .reservation_events
        .iter()
        .filter(|event| {
            event.outcome.as_str() == "granted"
                && event.reservation_kind == ReservationKind::FinalSort
                && event.memory_boundary == "final_sort"
        })
        .map(|event| event.requested_bytes)
        .collect::<Vec<_>>();
    assert_eq!(final_sort_bytes.len(), 2, "full final sort should reserve sort order and materialized output");
    let sequential_peak = input_bytes.saturating_add(final_sort_bytes[0]).saturating_add(final_sort_bytes[1]);
    let constrained_budget = sequential_peak - 1;
    assert!(
        constrained_budget > input_bytes,
        "test data must leave a budget window where source read fits but full final sort reservation fails"
    );

    let constrained_context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: constrained_budget,
            max_source_chunk_bytes: 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 3,
            min_retry_chunk_row_groups: 1,
        })
        .build()
        .unwrap();

    let result = execute_query_plan(&constrained_context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.source_chunks_planned, 1);
    assert_eq!(metrics.source_chunks_committed, 1);
    assert!(metrics.retry_count >= 1);
    assert!(
        metrics.retry_reservation_failure_count >= 1 || metrics.retry_resource_exhausted_count >= 1,
        "full final sort must fail through a retryable bounded-memory path"
    );
    assert_final_sort_prepare_budget_event(
        metrics,
        "cudf_query_engine.final_sort.prepare",
        "final_sort",
        Some(final_sort_bytes[0]),
    );
    assert_final_sort_prepare_budget_event(
        metrics,
        "cudf_query_engine.final_sort.topk_prepare",
        "final_sort_topk_chunk",
        None,
    );
    assert_final_sort_materialize_budget_event(
        metrics,
        "cudf_query_engine.final_sort.topk_materialize",
        "final_sort_topk_chunk",
        None,
    );
    assert_final_sort_prepare_budget_event(
        metrics,
        "cudf_query_engine.final_sort.topk_prepare",
        "final_sort_topk_merge",
        None,
    );
    assert_final_sort_materialize_budget_event(
        metrics,
        "cudf_query_engine.final_sort.topk_materialize",
        "final_sort_topk_merge",
        None,
    );
    assert!(metrics.reservation_events.iter().any(|event| {
        event.outcome.as_str() == "granted"
            && event.reservation_kind == ReservationKind::FinalSort
            && event.memory_boundary == "final_sort_output"
            && event.requested_bytes > 0
            && event.requested_bytes < final_sort_bytes[1]
    }));

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), expected.num_rows());
    assert_eq!(int32_array(batch, 0).values().as_ref(), int32_array(expected, 0).values().as_ref());
    assert_eq!(int64_array(batch, 1).values().as_ref(), int64_array(expected, 1).values().as_ref());

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_accounts_full_final_sort_reservation_during_limited_output() {
    let dir = unique_temp_dir("cudf-query-engine-bounded-final-sort-limit-reservation");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_single_row_group_final_sort_input_parquet(&path, 4096);
    let aggregated_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("sum_m00", DataType::Int64, true),
    ]));
    let final_sort_schema = wide_final_sort_schema(16);

    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["key".to_string(), "m00".to_string()],
    );
    let aggregate = AggregateSpec::new(
        vec![Expr::column("key")],
        vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("m00"), "sum_m00")],
    );
    let sort = SortSpec::new(vec![SortExpr::new(Expr::column("key"), true, false)]);

    let mut builder = QueryPlanBuilder::new();
    let source_id = builder.add_source(SourceNode::LocalParquet(source));
    let aggregate = builder.add_aggregate(source_id, aggregate, Arc::clone(&aggregated_schema));
    let projection = builder
        .add_projection(aggregate, projection_outputs(wide_final_sort_projection(16), Arc::clone(&final_sort_schema)));
    let sort = builder.add_sort(projection, sort, Arc::clone(&final_sort_schema));
    let limit = builder.add_limit(sort, LimitSpec::fetch(4095), Arc::clone(&final_sort_schema));
    let plan = builder.finish_with_host_sink(limit);

    let baseline_context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 8 * 1024 * 1024 * 1024,
            max_source_chunk_bytes: 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .build()
        .unwrap();
    let baseline = execute_query_plan(&baseline_context, &plan).unwrap();
    let baseline_metrics = baseline.metrics();
    let first_final_sort_event = baseline_metrics
        .reservation_events
        .iter()
        .position(|event| {
            event.outcome.as_str() == "granted"
                && event.reservation_kind == ReservationKind::FinalSort
                && event.memory_boundary == "final_sort"
        })
        .expect("baseline should reserve final sort work");
    let input_bytes = baseline_metrics.reservation_events[..first_final_sort_event]
        .iter()
        .rfind(|event| event.outcome.as_str() == "granted")
        .expect("baseline should have a resident aggregate input before final sort")
        .requested_bytes;
    let final_sort_bytes = baseline_metrics
        .reservation_events
        .iter()
        .filter(|event| {
            event.outcome.as_str() == "granted"
                && event.reservation_kind == ReservationKind::FinalSort
                && event.memory_boundary == "final_sort"
        })
        .map(|event| event.requested_bytes)
        .collect::<Vec<_>>();
    assert_eq!(final_sort_bytes.len(), 2, "full final sort should reserve sort order and materialized output");
    let sorted_order_bytes = final_sort_bytes[0];
    let materialized_output_bytes = final_sort_bytes[1];
    let limited_output_bytes =
        granted_reservation_requested_bytes(baseline_metrics, ReservationKind::FinalSort, "final_sort_output");

    let sequential_peak = input_bytes.saturating_add(sorted_order_bytes).saturating_add(materialized_output_bytes);
    let overlapping_peak = input_bytes.saturating_add(materialized_output_bytes).saturating_add(limited_output_bytes);
    assert!(
        overlapping_peak > sequential_peak + 1,
        "test data must create a budget window between sequential and overlapping final-sort reservations"
    );
    let constrained_budget = sequential_peak + ((overlapping_peak - sequential_peak) / 2);
    assert!(constrained_budget > sequential_peak);
    assert!(constrained_budget < overlapping_peak);

    let constrained_context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: constrained_budget,
            max_source_chunk_bytes: 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .build()
        .unwrap();

    let error = match execute_query_plan(&constrained_context, &plan) {
        Ok(_) => panic!("bounded execution should account full final-sort output while producing limited output"),
        Err(error) => error,
    };
    assert_eq!(error.code(), QueryErrorCode::MemoryReservationFailed);
    assert_eq!(error.fact_value("memory_boundary"), Some("final_sort_output"));
    assert_eq!(error.fact_value("reservation_kind"), Some("final_sort"));
    assert_eq!(error.fact_value("stage"), Some("final_sort"));
    assert_eq!(error.fact_value("attempt"), Some("1"));
    assert_eq!(error.fact_value("max_attempts"), Some("1"));
    assert_eq!(error.fact_value("retry_action"), Some("retry_attempts_exhausted"));
    assert_eq!(error.fact_value("resume_boundary"), Some("final_sort"));
    let constrained_budget = constrained_budget.to_string();
    let limited_output_bytes = limited_output_bytes.to_string();
    assert_eq!(error.fact_value("budget_bytes"), Some(constrained_budget.as_str()));
    assert_eq!(error.fact_value("requested_bytes"), Some(limited_output_bytes.as_str()));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_reports_retry_facts_when_final_sort_without_limit_fails() {
    let dir = unique_temp_dir("cudf-query-engine-bounded-final-sort-retry-exhausted");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_single_row_group_final_sort_input_parquet(&path, 4096);
    let aggregated_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("sum_m00", DataType::Int64, true),
    ]));
    let final_sort_schema = wide_final_sort_schema(16);

    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["key".to_string(), "m00".to_string()],
    );
    let aggregate = AggregateSpec::new(
        vec![Expr::column("key")],
        vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("m00"), "sum_m00")],
    );
    let sort = SortSpec::new(vec![SortExpr::new(Expr::column("key"), true, false)]);

    let mut builder = QueryPlanBuilder::new();
    let source_id = builder.add_source(SourceNode::LocalParquet(source));
    let aggregate = builder.add_aggregate(source_id, aggregate, Arc::clone(&aggregated_schema));
    let projection = builder
        .add_projection(aggregate, projection_outputs(wide_final_sort_projection(16), Arc::clone(&final_sort_schema)));
    let sort = builder.add_sort(projection, sort, final_sort_schema);
    let plan = builder.finish_with_host_sink(sort);

    let baseline_context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 8 * 1024 * 1024 * 1024,
            max_source_chunk_bytes: 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .build()
        .unwrap();
    let baseline = execute_query_plan(&baseline_context, &plan).unwrap();
    let baseline_metrics = baseline.metrics();
    let first_final_sort_event = baseline_metrics
        .reservation_events
        .iter()
        .position(|event| {
            event.outcome.as_str() == "granted"
                && event.reservation_kind == ReservationKind::FinalSort
                && event.memory_boundary == "final_sort"
        })
        .expect("baseline should reserve final sort work");
    let input_bytes = baseline_metrics.reservation_events[..first_final_sort_event]
        .iter()
        .rfind(|event| event.outcome.as_str() == "granted")
        .expect("baseline should have a resident aggregate input before final sort")
        .requested_bytes;
    let final_sort_bytes = baseline_metrics
        .reservation_events
        .iter()
        .filter(|event| {
            event.outcome.as_str() == "granted"
                && event.reservation_kind == ReservationKind::FinalSort
                && event.memory_boundary == "final_sort"
        })
        .map(|event| event.requested_bytes)
        .collect::<Vec<_>>();
    assert_eq!(final_sort_bytes.len(), 2, "full final sort should reserve sort order and materialized output");
    let sequential_peak = input_bytes.saturating_add(final_sort_bytes[0]).saturating_add(final_sort_bytes[1]);
    let constrained_budget = sequential_peak - 1;

    let constrained_context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: constrained_budget,
            max_source_chunk_bytes: 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 3,
            min_retry_chunk_row_groups: 1,
        })
        .build()
        .unwrap();

    let error = match execute_query_plan(&constrained_context, &plan) {
        Ok(_) => panic!("bounded execution should fail final sort when no top-k retry boundary exists"),
        Err(error) => error,
    };

    assert!(matches!(error.code(), QueryErrorCode::MemoryReservationFailed | QueryErrorCode::CudfResourceExhausted));
    assert_eq!(error.fact_value("memory_boundary"), Some("final_sort"));
    assert_eq!(error.fact_value("stage"), Some("final_sort"));
    assert_eq!(error.fact_value("attempt"), Some("1"));
    assert_eq!(error.fact_value("max_attempts"), Some("3"));
    assert_eq!(error.fact_value("retry_action"), Some("retry_attempts_exhausted"));
    assert_eq!(error.fact_value("resume_boundary"), Some("final_sort"));
    assert_eq!(error.fact_value("committed_chunks"), Some("1"));
    assert_eq!(error.fact_value("committed_partials"), Some("1"));
    if error.code() == QueryErrorCode::MemoryReservationFailed {
        assert_eq!(error.fact_value("reservation_kind"), Some("final_sort"));
    } else {
        assert_eq!(error.fact_value("budget_boundary"), Some("final_sort"));
        assert_eq!(error.fact_value("budget_enforcement"), Some("hard_memory_resource_limit"));
    }

    fs::remove_dir_all(dir).unwrap();
}
