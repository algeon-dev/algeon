use super::*;

#[test]
fn bounded_native_query_executes_chunked_grouped_sum_and_final_merge() {
    let dir = unique_temp_dir("cudf-query-engine-bounded-native-execution");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_row_grouped_input_parquet(&path);

    let projected_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("x00", DataType::Int64, false),
    ]));
    let aggregated_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("sum_x00", DataType::Int64, true),
    ]));

    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["key".to_string(), "m00".to_string(), "m01".to_string(), "payload".to_string()],
    );
    let filter = Expr::binary(Expr::column("payload"), BinaryOp::GtEq, Expr::literal(LiteralValue::Int64(10)));
    let projection =
        vec![Expr::column("key"), Expr::binary(Expr::column("m00"), BinaryOp::Add, Expr::column("m01")).alias("x00")];
    let aggregate = AggregateSpec::new(
        vec![Expr::column("key")],
        vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("x00"), "sum_x00")],
    );
    let sort = SortSpec::new(vec![
        SortExpr::new(Expr::column("sum_x00"), true, false),
        SortExpr::new(Expr::column("key"), false, false),
    ]);

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let filter = builder.add_filter(source, filter, Arc::clone(&input_schema));
    let projection = builder.add_projection(filter, projection_outputs(projection, projected_schema));
    let aggregate = builder.add_aggregate(projection, aggregate, Arc::clone(&aggregated_schema));
    let sort = builder.add_sort(aggregate, sort, Arc::clone(&aggregated_schema));
    let limit = builder.add_limit(sort, LimitSpec::fetch(3), aggregated_schema);
    let plan = builder.finish_with_host_sink(limit);

    let device_budget_bytes = 8 * 1024 * 1024 * 1024;
    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes,
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
    assert_eq!(metrics.phase_c_memory_control_policy.as_str(), "bounded");
    assert_eq!(metrics.source_chunks_planned, 3);
    assert_eq!(metrics.source_chunks_attempted, 3);
    assert_eq!(metrics.source_chunks_committed, 3);
    assert_eq!(metrics.source_read_prefetches_started, 0);
    assert_eq!(metrics.source_read_prefetches_completed, 0);
    assert_eq!(metrics.source_read_prefetches_skipped, 0);
    assert_eq!(metrics.source_read_prefetches_abandoned, 0);
    assert!(
        metrics.reservation_events.iter().all(|event| {
            !(event.memory_boundary == "source_chunk_read" && event.action == MemoryReservationEventAction::Grow)
        }),
        "source-adjacent aggregate filtering/projection should not grow raw source-read reservations to observed bytes"
    );
    assert!(
        metrics.reservation_events.iter().any(|event| {
            event.memory_boundary == "source_chunk_filter_output"
                && event.action == MemoryReservationEventAction::Transfer
        }),
        "source-adjacent aggregate filtering should transfer ownership to the filtered output boundary"
    );
    assert!(
        metrics.reservation_events.iter().any(|event| {
            event.memory_boundary == "source_chunk_projection_output"
                && event.action == MemoryReservationEventAction::Transfer
        }),
        "source-adjacent aggregate projection should transfer ownership to the projected output boundary"
    );
    let first_chunk_operator = metrics
        .reservation_events
        .iter()
        .position(|event| event.outcome.as_str() == "granted" && event.memory_boundary == "chunk_operator")
        .expect("chunk operator reservation should be recorded");
    let prefetches_before_first_operator = metrics.reservation_events[..first_chunk_operator]
        .iter()
        .filter(|event| event.outcome.as_str() == "granted" && event.memory_boundary == "source_chunk_prefetch")
        .count();
    assert_eq!(prefetches_before_first_operator, 0);
    assert!(
        metrics.reservation_events.iter().all(|event| event.memory_boundary != "source_chunk_prefetch"),
        "source-adjacent aggregate chunks should not prefetch while the transformed source output and chunk operator reservation overlap"
    );
    assert_eq!(metrics.row_groups_planned, 3);
    assert_eq!(metrics.row_groups_committed, 3);
    assert_eq!(metrics.partial_aggregate_count, 3);
    assert!(metrics.partial_aggregate_input_rows >= metrics.partial_aggregate_rows);
    assert!(metrics.partial_aggregate_retained_bytes > 0);
    assert_eq!(metrics.final_merge_attempts, 1);
    assert_eq!(metrics.final_merge_levels, 1);
    assert_eq!(metrics.final_merge_max_fan_in, 3);
    assert!(metrics.task_operator_metrics.contains_key("source_read"));
    assert!(metrics.task_operator_metrics.contains_key("unary"));
    assert!(metrics.task_operator_metrics.contains_key("aggregate"));
    assert!(metrics.task_operator_metrics.contains_key("final_merge"));
    assert!(metrics.task_operator_metrics["source_read"].reservation_requested_bytes > 0);
    assert!(metrics.task_operator_metrics["aggregate"].reservation_requested_bytes > 0);
    assert!(metrics.max_task_reservation_bytes > 0);
    assert!(
        metrics.reservation_live_peak_bytes < device_budget_bytes,
        "source-adjacent aggregate work reservations should not saturate the configured device budget"
    );
    assert!(metrics.reservation_events.iter().filter(|event| event.outcome.as_str() == "denied").all(|event| {
        event.stage.is_some()
            && event.compiled_task_id.as_deref().is_some_and(|task_id| task_id.starts_with("legacy_bounded."))
    }));
    assert_eq!(metrics.parquet_files, 1);
    assert_eq!(metrics.row_groups_selected, 3);
    assert_eq!(metrics.read_columns, 4);
    assert_eq!(metrics.input_batches, 3);
    assert_eq!(metrics.input_rows, 6);
    assert_eq!(metrics.boundary.host_to_device_uploads, 0);
    assert_eq!(metrics.boundary.device_to_host_materializations, 1);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 3);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[2, 1, 3]);
    assert_eq!(int64_array(batch, 1).values().as_ref(), &[707, 404, 404]);
    let source_events = budgeted_events(metrics, "cudf_query_engine.local_parquet_chunk.read");
    assert_eq!(source_events.len(), 3);
    let selected_row_groups =
        source_events.iter().map(|event| event.context_facts["selected_row_groups"].as_str()).collect::<Vec<_>>();
    assert_eq!(selected_row_groups, vec!["0", "1", "2"]);
    for event in source_events {
        assert_source_budget_event_context(event, "key,m00,m01,payload", "key,m00,m01,payload");
        assert_eq!(event.context_facts.get("reads_all_row_groups").map(String::as_str), Some("false"));
        assert_eq!(event.context_facts.get("selected_row_group_count").map(String::as_str), Some("1"));
    }

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_uses_configured_source_cap_when_budget_model_allows_it() {
    let dir = unique_temp_dir("cudf-query-engine-bounded-aggregate-source-refinement");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_row_grouped_input_parquet(&path);

    let aggregated_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("sum_m00", DataType::Int64, true),
    ]));

    let source = LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new(&path)
                .with_file_size_bytes(fs::metadata(&path).unwrap().len())
                .with_row_group_bytes(vec![350, 350, 350])
                .with_projected_row_group_bytes(vec![350, 350, 350], vec![350, 350, 350])
                .with_row_group_rows(vec![2, 2, 2]),
        ],
        Arc::clone(&input_schema),
        vec!["key".to_string(), "m00".to_string()],
    );
    let aggregate = AggregateSpec::new(
        vec![Expr::column("key")],
        vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("m00"), "sum_m00")],
    );

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let aggregate = builder.add_aggregate(source, aggregate, Arc::clone(&aggregated_schema));
    let plan = builder.finish_with_host_sink(aggregate);

    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 8 * 1024 * 1024 * 1024,
            max_source_chunk_bytes: 1600,
            max_row_groups_per_chunk: 8,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .build()
        .unwrap();
    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.source_chunks_planned, 1);
    assert_eq!(metrics.source_chunks_retried, 0);
    assert_eq!(metrics.reservation_denied_count, 0);
    assert_eq!(metrics.partial_aggregate_count, 1);
    assert!(
        metrics
            .reservation_events
            .iter()
            .all(|event| { !(event.memory_boundary == "source_chunk_read" && event.requested_bytes > 1600) })
    );

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 3);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_executes_min_max_count_and_final_merge() {
    let dir = unique_temp_dir("cudf-query-engine-bounded-native-aggregate-parity");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_row_grouped_input_parquet(&path);

    let aggregated_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("min_m00", DataType::Int64, true),
        Field::new("max_m01", DataType::Int64, true),
        Field::new("count_payload", DataType::Int64, false),
    ]));

    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["key".to_string(), "payload".to_string(), "m00".to_string(), "m01".to_string()],
    );
    let filter = Expr::binary(Expr::column("payload"), BinaryOp::GtEq, Expr::literal(LiteralValue::Int64(10)));
    let aggregate = AggregateSpec::new(
        vec![Expr::column("key")],
        vec![
            AggregateExpr::new(AggregateFunction::Min, Expr::column("m00"), "min_m00"),
            AggregateExpr::new(AggregateFunction::Max, Expr::column("m01"), "max_m01"),
            AggregateExpr::new(AggregateFunction::Count, Expr::column("payload"), "count_payload"),
        ],
    );
    let sort = SortSpec::new(vec![SortExpr::new(Expr::column("key"), false, false)]);

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let filter = builder.add_filter(source, filter, Arc::clone(&input_schema));
    let aggregate = builder.add_aggregate(filter, aggregate, Arc::clone(&aggregated_schema));
    let sort = builder.add_sort(aggregate, sort, Arc::clone(&aggregated_schema));
    let limit = builder.add_limit(sort, LimitSpec::fetch(3), aggregated_schema);
    let plan = builder.finish_with_host_sink(limit);

    let context = ExecutionContext::builder()
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
    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.source_chunks_planned, 3);
    assert_eq!(metrics.source_chunks_committed, 3);
    assert_eq!(metrics.partial_aggregate_count, 3);
    assert!(metrics.partial_aggregate_input_rows >= metrics.partial_aggregate_rows);
    assert!(metrics.partial_aggregate_retained_bytes > 0);
    assert_eq!(metrics.final_merge_attempts, 1);
    assert_eq!(metrics.final_merge_max_fan_in, 3);
    assert_eq!(metrics.boundary.host_to_device_uploads, 0);
    assert_eq!(metrics.boundary.device_to_host_materializations, 1);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 3);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[1, 2, 3]);
    assert_eq!(int64_array(batch, 1).values().as_ref(), &[100, 200, 400]);
    assert_eq!(int64_array(batch, 2).values().as_ref(), &[3, 5, 4]);
    assert_eq!(int64_array(batch, 3).values().as_ref(), &[2, 2, 1]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_casts_single_partial_grouped_count_to_output_schema() {
    let dir = unique_temp_dir("cudf-query-engine-bounded-native-single-partial-count");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_row_grouped_input_parquet(&path);

    let aggregated_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("count_payload", DataType::Int64, false),
    ]));

    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["key".to_string(), "payload".to_string()],
    );
    let filter = Expr::binary(Expr::column("payload"), BinaryOp::GtEq, Expr::literal(LiteralValue::Int64(10)));
    let aggregate = AggregateSpec::new(
        vec![Expr::column("key")],
        vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("payload"), "count_payload")],
    );
    let sort = SortSpec::new(vec![SortExpr::new(Expr::column("key"), false, false)]);

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let filter = builder.add_filter(source, filter, Arc::clone(&input_schema));
    let aggregate = builder.add_aggregate(filter, aggregate, Arc::clone(&aggregated_schema));
    let sort = builder.add_sort(aggregate, sort, Arc::clone(&aggregated_schema));
    let plan = builder.finish_with_host_sink(sort);

    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 8 * 1024 * 1024 * 1024,
            max_source_chunk_bytes: 8 * 1024 * 1024 * 1024,
            max_row_groups_per_chunk: 16,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .build()
        .unwrap();
    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.source_chunks_planned, 1);
    assert_eq!(metrics.partial_aggregate_count, 1);
    assert_eq!(metrics.final_merge_attempts, 0);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.column(1).data_type(), &DataType::Int64);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[1, 2, 3]);
    assert_eq!(int64_array(batch, 1).values().as_ref(), &[2, 2, 1]);

    fs::remove_dir_all(dir).unwrap();
}
