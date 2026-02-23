use super::*;

#[test]
fn encoded_key_partial_contract_decodes_grouped_string_keys_before_merge() {
    let runtime = cudf::Runtime::new().unwrap();
    let input = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("key", DataType::Utf8, true),
            Field::new("value", DataType::Int64, true),
        ])),
        vec![
            Arc::new(StringArray::from(vec![
                Some("segment-alpha-long"),
                None,
                Some("segment-alpha-long"),
                Some("segment-beta-long"),
                Some("segment-beta-long"),
                Some("segment-beta-long"),
                None,
            ])),
            Arc::new(Int64Array::from(vec![Some(10), Some(20), Some(30), Some(40), None, Some(50), Some(5)])),
        ],
    )
    .unwrap();
    let frame = DataFrame::from_record_batch(&runtime, &input).unwrap();
    let groupby_options = cudf::GroupByOptions::builder().include_null_keys(true).build();
    let aggregate_specs = [
        cudf::AggSpec::new("value", cudf::GroupByAgg::Sum).with_alias("sum_value"),
        cudf::AggSpec::new("value", cudf::GroupByAgg::Count { include_nulls: false }).with_alias("count_value"),
    ];

    let direct = frame.groupby(["key"]).options(groupby_options.clone()).agg(&aggregate_specs).unwrap();

    let codebook = frame.codebook("key").unwrap();
    let encoded = frame.with_encoded_column("__encoded_key", "key", &codebook).unwrap();
    let encoded_grouped = encoded.groupby(["__encoded_key"]).options(groupby_options).agg(&aggregate_specs).unwrap();
    let encoded_grouped_batch = encoded_grouped.to_record_batch().unwrap();
    assert!(
        (0..encoded_grouped_batch.num_rows()).any(|row| encoded_grouped_batch.column(0).is_null(row)),
        "encoded groupby must retain a null-code group when include_null_keys=true"
    );

    let decoded = encoded_grouped
        .with_decoded_column("key", "__encoded_key", &codebook)
        .unwrap()
        .select(["key", "sum_value", "count_value"])
        .unwrap();

    assert_eq!(direct.column_names(), vec!["key", "sum_value", "count_value"]);
    assert_eq!(decoded.column_names(), vec!["key", "sum_value", "count_value"]);
    assert_eq!(grouped_string_rows(&decoded), grouped_string_rows(&direct));
}

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
    assert!(
        metrics.reservation_events.iter().all(|event| event.memory_boundary != "source_chunk_prefetch"),
        "source-adjacent aggregate chunks should not prefetch while transformed source output is partitioned"
    );
    assert_eq!(metrics.row_groups_planned, 3);
    assert_eq!(metrics.row_groups_committed, 3);
    assert_eq!(metrics.partial_aggregate_count, 3);
    assert_eq!(metrics.partial_aggregate_input_rows, 5);
    assert!(metrics.partial_aggregate_rows >= 3);
    assert!(metrics.partial_aggregate_retained_bytes > 0);
    assert!(metrics.chunk_operator_time.as_nanos() > 0);
    assert_eq!(metrics.final_merge_attempts, 1);
    assert_eq!(metrics.final_merge_levels, 1);
    assert!(metrics.final_merge_max_fan_in >= 1);
    assert!(metrics.task_operator_metrics.contains_key("source_read"));
    assert!(metrics.task_operator_metrics.contains_key("unary"));
    assert!(metrics.task_operator_metrics.contains_key("aggregate"));
    assert!(metrics.task_operator_metrics["source_read"].reservation_requested_bytes > 0);
    assert!(metrics.task_operator_metrics["aggregate"].reservation_requested_bytes > 0);
    assert!(metrics.max_task_reservation_bytes > 0);
    for boundary in [
        "source_adjacent_aggregate_hash_partition",
        "source_adjacent_aggregate_partition",
        "source_adjacent_aggregate_partition_concat",
        "source_adjacent_aggregate_partition_concat_output",
        "source_adjacent_partitioned_partial_aggregate",
        "source_adjacent_partitioned_partial_aggregate_output",
        "final_aggregate_hash_partition",
        "final_aggregate_partition",
        "partitioned_final_aggregate_merge_output",
    ] {
        assert!(
            metrics.reservation_events.iter().any(|event| event.memory_boundary == boundary),
            "missing source-adjacent partitioned aggregate reservation boundary {boundary}"
        );
    }
    let partial_strategy_rows = metrics
        .aggregate_strategy_decisions
        .iter()
        .filter(|decision| decision.facts.get("execution_stage").map(String::as_str) == Some("partial"))
        .collect::<Vec<_>>();
    assert_eq!(partial_strategy_rows.len(), 3);
    let mut diagnostic_input_rows = 0u64;
    for partial_strategy in &partial_strategy_rows {
        assert_eq!(partial_strategy.node_id, aggregate.to_string());
        assert_eq!(partial_strategy.facts.get("selected_strategy").map(String::as_str), Some("libcudf_groupby"));
        assert_eq!(partial_strategy.facts.get("memory_mode").map(String::as_str), Some("bounded"));
        assert_eq!(partial_strategy.facts.get("group_key_count").map(String::as_str), Some("1"));
        assert_eq!(partial_strategy.facts.get("group_key_dtypes").map(String::as_str), Some("Int32"));
        assert_eq!(partial_strategy.facts.get("aggregate_functions").map(String::as_str), Some("sum"));
        assert_eq!(partial_strategy.facts.get("aggregate_filter_count").map(String::as_str), Some("0"));
        assert_eq!(partial_strategy.facts.get("distinct_aggregate_count").map(String::as_str), Some("0"));
        assert_eq!(partial_strategy.facts.get("chunk_count").map(String::as_str), Some("3"));
        let chunk_row_count = partial_strategy
            .facts
            .get("chunk_row_count")
            .and_then(|value| value.parse::<u64>().ok())
            .expect("partial row should record row-balanced partition input rows");
        diagnostic_input_rows = diagnostic_input_rows.saturating_add(chunk_row_count);
        assert_eq!(partial_strategy.facts.get("source_adjacent_partial").map(String::as_str), Some("true"));
        assert_eq!(partial_strategy.facts.get("source_chunk_row_group_count").map(String::as_str), Some("3"));
        assert_eq!(partial_strategy.facts.get("source_chunk_known_row_count").map(String::as_str), Some("6"));
        assert_eq!(partial_strategy.facts.get("encode_elapsed_ns").map(String::as_str), Some("0"));
        assert_eq!(partial_strategy.facts.get("decode_elapsed_ns").map(String::as_str), Some("0"));
        assert_eq!(
            partial_strategy.facts.get("rejected_candidate_strategies").map(String::as_str),
            Some(
                "encoded_key_partial:encoded_key_partial_requires_string_group_key;prepared_sort_groupby:prepared_sort_groupby_requires_runtime_cardinality_evidence;partitioned_merge:partitioned_merge_requires_final_merge_stage"
            )
        );
        assert!(
            partial_strategy
                .facts
                .get("groupby_elapsed_ns")
                .and_then(|value| value.parse::<u64>().ok())
                .is_some_and(|value| value > 0)
        );
    }
    assert_eq!(diagnostic_input_rows, metrics.partial_aggregate_input_rows);
    let final_strategy = metrics
        .aggregate_strategy_decisions
        .iter()
        .find(|decision| decision.facts.get("execution_stage").map(String::as_str) == Some("final_merge"))
        .expect("row-balanced source-adjacent aggregate should record final merge strategy");
    assert_eq!(final_strategy.node_id, aggregate.to_string());
    assert_eq!(final_strategy.facts.get("selected_strategy").map(String::as_str), Some("partitioned_merge"));
    assert_eq!(final_strategy.facts.get("partial_aggregate_count").map(String::as_str), Some("3"));
    assert_eq!(final_strategy.facts.get("partitioned_merge_contract_status").map(String::as_str), Some("available"));
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
fn bounded_native_query_executes_source_adjacent_grouped_sum_with_empty_partitions() {
    let dir = unique_temp_dir("cudf-query-engine-bounded-source-adjacent-empty-partitions");
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

    let large_row_group_bytes = 900 * 1024 * 1024;
    let source = LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new(&path)
                .with_file_size_bytes(fs::metadata(&path).unwrap().len())
                .with_row_group_bytes(vec![large_row_group_bytes; 3])
                .with_projected_row_group_bytes(vec![large_row_group_bytes; 3], vec![large_row_group_bytes; 3])
                .with_row_group_rows(vec![2, 2, 2]),
        ],
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

    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 8 * 1024 * 1024 * 1024,
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
    assert_eq!(metrics.source_chunks_planned, 3);
    assert_eq!(metrics.source_chunks_committed, 3);
    assert!(
        metrics.partial_aggregate_count > 5,
        "large source estimates should force more row-balanced partitions than non-empty input rows"
    );
    assert_eq!(metrics.final_merge_attempts, 1);
    assert!(metrics.final_merge_max_fan_in >= 1);
    assert!(
        metrics
            .reservation_events
            .iter()
            .any(|event| { event.memory_boundary == "source_adjacent_partitioned_partial_aggregate_output" })
    );
    assert!(
        metrics
            .reservation_events
            .iter()
            .any(|event| event.memory_boundary == "partitioned_final_aggregate_merge_output")
    );

    let partial_strategy_rows = metrics
        .aggregate_strategy_decisions
        .iter()
        .filter(|decision| decision.facts.get("execution_stage").map(String::as_str) == Some("partial"))
        .collect::<Vec<_>>();
    assert_eq!(partial_strategy_rows.len(), metrics.partial_aggregate_count as usize);
    let final_strategy = metrics
        .aggregate_strategy_decisions
        .iter()
        .find(|decision| decision.facts.get("execution_stage").map(String::as_str) == Some("final_merge"))
        .expect("multi-partition source-adjacent aggregate should record final merge strategy");
    assert_eq!(final_strategy.node_id, aggregate.to_string());
    assert_eq!(final_strategy.facts.get("selected_strategy").map(String::as_str), Some("partitioned_merge"));
    assert_eq!(final_strategy.facts.get("partitioned_merge_contract_status").map(String::as_str), Some("available"));

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 3);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[2, 1, 3]);
    assert_eq!(int64_array(batch, 1).values().as_ref(), &[707, 404, 404]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_uses_group_key_source_adjacent_partitions_for_string_keys() {
    let dir = unique_temp_dir("cudf-query-engine-bounded-source-adjacent-group-key-partitions");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_nullable_string_grouped_aggregate_parquet(&path);

    let projected_schema = Arc::new(Schema::new(vec![
        Field::new("group_key", DataType::Utf8, true),
        Field::new("value", DataType::Int64, true),
    ]));
    let aggregated_schema = Arc::new(Schema::new(vec![
        Field::new("group_key", DataType::Utf8, true),
        Field::new("sum_value", DataType::Int64, true),
        Field::new("count_value", DataType::Int64, false),
        Field::new("min_value", DataType::Int64, true),
        Field::new("max_value", DataType::Int64, true),
    ]));

    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["key".to_string(), "value".to_string()],
    );
    let projection = vec![Expr::column("key").alias("group_key"), Expr::column("value")];
    let aggregate = AggregateSpec::new(
        vec![Expr::column("group_key")],
        vec![
            AggregateExpr::new(AggregateFunction::Sum, Expr::column("value"), "sum_value"),
            AggregateExpr::new(AggregateFunction::Count, Expr::column("value"), "count_value"),
            AggregateExpr::new(AggregateFunction::Min, Expr::column("value"), "min_value"),
            AggregateExpr::new(AggregateFunction::Max, Expr::column("value"), "max_value"),
        ],
    );

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let projection = builder.add_projection(source, projection_outputs(projection, projected_schema));
    let aggregate = builder.add_aggregate(projection, aggregate, Arc::clone(&aggregated_schema));
    let plan = builder.finish_with_host_sink(aggregate);

    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 8 * 1024 * 1024 * 1024,
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
    assert_eq!(metrics.source_chunks_planned, 3);
    assert_eq!(metrics.source_chunks_committed, 3);
    assert!(metrics.partial_aggregate_count > 0);

    let partial_strategy_rows = metrics
        .aggregate_strategy_decisions
        .iter()
        .filter(|decision| decision.facts.get("execution_stage").map(String::as_str) == Some("partial"))
        .collect::<Vec<_>>();
    assert_eq!(partial_strategy_rows.len(), metrics.partial_aggregate_count as usize);
    let first_partial = partial_strategy_rows.first().expect("group-key partitioned partial strategy rows");
    assert_eq!(first_partial.node_id, aggregate.to_string());
    assert_eq!(first_partial.facts.get("source_adjacent_partition_strategy").map(String::as_str), Some("group_key"));
    assert_eq!(first_partial.facts.get("source_adjacent_partition_key_count").map(String::as_str), Some("1"));
    let planned_partitions = first_partial
        .facts
        .get("source_adjacent_partition_count")
        .and_then(|value| value.parse::<u64>().ok())
        .expect("partition count fact");
    let non_empty_partitions = first_partial
        .facts
        .get("source_adjacent_non_empty_partition_count")
        .and_then(|value| value.parse::<u64>().ok())
        .expect("non-empty partition count fact");
    assert!(planned_partitions >= metrics.partial_aggregate_count);
    assert!(non_empty_partitions <= metrics.partial_aggregate_count);
    assert!(partial_strategy_rows.iter().all(|decision| {
        decision.facts.get("source_adjacent_partition_strategy").map(String::as_str) == Some("group_key")
            && decision.facts.get("source_chunk_row_group_count").map(String::as_str) == Some("3")
            && decision.facts.get("source_chunk_known_row_count").map(String::as_str) == Some("120")
    }));
    let prepared_partial = partial_strategy_rows
        .iter()
        .find(|decision| decision.facts.get("selected_strategy").map(String::as_str) == Some("prepared_sort_groupby"))
        .expect("low-cardinality source-adjacent partial should select prepared sort groupby after prior evidence");
    assert_eq!(prepared_partial.facts.get("prepared_sort_groupby_rejected_reason").map(String::as_str), Some("none"));
    assert_eq!(
        prepared_partial.facts.get("prepared_sort_groupby_contract_status").map(String::as_str),
        Some("available")
    );
    assert_eq!(
        prepared_partial.facts.get("prepared_sort_groupby_cost_model_status").map(String::as_str),
        Some("selected_source_adjacent_execution")
    );
    let prepared_num_groups = prepared_partial
        .facts
        .get("prepared_sort_groupby_num_groups")
        .and_then(|value| value.parse::<u64>().ok())
        .expect("prepared selected row should record exact group count");
    let prepared_key_bytes = prepared_partial
        .facts
        .get("prepared_sort_groupby_grouped_key_output_bytes")
        .and_then(|value| value.parse::<u64>().ok())
        .expect("prepared selected row should record grouped-key output bytes");
    let prepared_max_group_size = prepared_partial
        .facts
        .get("prepared_sort_groupby_max_group_size")
        .and_then(|value| value.parse::<u64>().ok())
        .expect("prepared selected row should record max group size");
    assert!(prepared_num_groups > 0);
    assert!(prepared_key_bytes > 0);
    assert!(prepared_max_group_size > 0);
    assert!(
        !prepared_partial
            .facts
            .get("rejected_candidate_strategies")
            .is_some_and(|strategies| strategies.contains("prepared_sort_groupby:")),
        "selected prepared strategy should not remain in the rejected-candidate summary"
    );
    let prepare_events = budgeted_events(metrics, "cudf_query_engine.aggregate.prepare_partial.prepared_sort_groupby");
    assert!(!prepare_events.is_empty(), "prepared strategy should record a budgeted prepare event");
    assert!(prepare_events.iter().any(|event| {
        event.unresolved_contract.as_deref() == Some("libcudf_prepared_groupby_retained_state_memory_unavailable")
            && event.output_rows == Some(prepared_num_groups)
            && event.output_bytes.is_none()
            && event.retained_bytes.is_none()
            && event.context_facts.get("prepared_groupby_strategy").map(String::as_str) == Some("prepared_sort_groupby")
    }));
    let aggregate_events =
        budgeted_events(metrics, "cudf_query_engine.aggregate.execute_partial.prepared_sort_groupby");
    assert!(!aggregate_events.is_empty(), "prepared strategy should record a budgeted aggregate event");
    let expected_prepared_num_groups = prepared_num_groups.to_string();
    assert!(aggregate_events.iter().any(|event| {
        event.unresolved_contract.as_deref()
            == Some("libcudf_groupby_aggregate_temporary_workspace_preflight_unavailable")
            && event.output_rows.is_some_and(|rows| rows > 0)
            && event.output_bytes.is_some_and(|bytes| bytes > 0)
            && event.retained_bytes.is_some_and(|bytes| bytes > 0)
            && event.context_facts.get("prepared_groupby_num_groups").map(String::as_str)
                == Some(expected_prepared_num_groups.as_str())
    }));

    let final_strategy = metrics
        .aggregate_strategy_decisions
        .iter()
        .find(|decision| decision.facts.get("execution_stage").map(String::as_str) == Some("final_merge"))
        .expect("group-key source-adjacent aggregate should record final merge strategy");
    assert_eq!(final_strategy.node_id, aggregate.to_string());
    assert_eq!(final_strategy.facts.get("selected_strategy").map(String::as_str), Some("partitioned_merge"));

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(
        grouped_string_sum_count_min_max_rows(batch),
        vec![
            (None, Some(150), 30, Some(5), Some(5)),
            (Some("segment-alpha-long".to_string()), Some(300), 30, Some(10), Some(10)),
            (Some("segment-beta-long".to_string()), Some(60), 60, Some(1), Some(1)),
        ]
    );

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_keeps_large_string_source_adjacent_inputs_on_row_balanced_partitions() {
    let dir = unique_temp_dir("cudf-query-engine-bounded-source-adjacent-large-string-row-balanced");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_nullable_string_grouped_aggregate_parquet(&path);

    let projected_schema = Arc::new(Schema::new(vec![
        Field::new("group_key", DataType::Utf8, true),
        Field::new("value", DataType::Int64, true),
    ]));
    let aggregated_schema = Arc::new(Schema::new(vec![
        Field::new("group_key", DataType::Utf8, true),
        Field::new("sum_value", DataType::Int64, true),
        Field::new("count_value", DataType::Int64, false),
    ]));

    let large_row_group_bytes = 900 * 1024 * 1024;
    let source = LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new(&path)
                .with_file_size_bytes(fs::metadata(&path).unwrap().len())
                .with_row_group_bytes(vec![large_row_group_bytes; 3])
                .with_projected_row_group_bytes(vec![large_row_group_bytes; 3], vec![large_row_group_bytes; 3])
                .with_row_group_rows(vec![40, 40, 40]),
        ],
        Arc::clone(&input_schema),
        vec!["key".to_string(), "value".to_string()],
    );
    let projection = vec![Expr::column("key").alias("group_key"), Expr::column("value")];
    let aggregate = AggregateSpec::new(
        vec![Expr::column("group_key")],
        vec![
            AggregateExpr::new(AggregateFunction::Sum, Expr::column("value"), "sum_value"),
            AggregateExpr::new(AggregateFunction::Count, Expr::column("value"), "count_value"),
        ],
    );

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let projection = builder.add_projection(source, projection_outputs(projection, projected_schema));
    let aggregate = builder.add_aggregate(projection, aggregate, Arc::clone(&aggregated_schema));
    let plan = builder.finish_with_host_sink(aggregate);

    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 8 * 1024 * 1024 * 1024,
            max_source_chunk_bytes: 1024 * 1024,
            max_row_groups_per_chunk: 8,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .with_source_scheduler_policy(SourceSchedulerPolicy::new(3))
        .build()
        .unwrap();
    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert!(
        metrics.partial_aggregate_count > 5,
        "large estimated source-adjacent string input should keep the bounded row-balanced partitioner"
    );
    let partial_strategy = metrics
        .aggregate_strategy_decisions
        .iter()
        .find(|decision| decision.facts.get("execution_stage").map(String::as_str) == Some("partial"))
        .expect("row-balanced partial strategy row");
    assert_eq!(
        partial_strategy.facts.get("source_adjacent_partition_strategy").map(String::as_str),
        Some("row_balanced")
    );

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    let mut rows = grouped_string_rows_from_batch(batch);
    rows.sort();
    assert_eq!(
        rows,
        vec![
            (None, Some(150), 30),
            (Some("segment-alpha-long".to_string()), Some(300), 30),
            (Some("segment-beta-long".to_string()), Some(60), 60),
        ]
    );

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_decodes_encoded_string_partials_before_final_merge() {
    let dir = unique_temp_dir("cudf-query-engine-bounded-encoded-string-aggregate");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_nullable_string_grouped_aggregate_parquet(&path);

    let aggregated_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Utf8, true),
        Field::new("sum_value", DataType::Int64, true),
        Field::new("count_value", DataType::Int64, false),
        Field::new("min_value", DataType::Int64, true),
        Field::new("max_value", DataType::Int64, true),
    ]));

    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["key".to_string(), "value".to_string()],
    );
    let aggregate = AggregateSpec::new(
        vec![Expr::column("key")],
        vec![
            AggregateExpr::new(AggregateFunction::Sum, Expr::column("value"), "sum_value"),
            AggregateExpr::new(AggregateFunction::Count, Expr::column("value"), "count_value"),
            AggregateExpr::new(AggregateFunction::Min, Expr::column("value"), "min_value"),
            AggregateExpr::new(AggregateFunction::Max, Expr::column("value"), "max_value"),
        ],
    );

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let aggregate = builder.add_aggregate(source, aggregate, Arc::clone(&aggregated_schema));
    let plan = builder.finish_with_host_sink(aggregate);

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
    assert_eq!(metrics.final_merge_attempts, 1);

    let partial_strategy_rows = metrics
        .aggregate_strategy_decisions
        .iter()
        .filter(|decision| decision.facts.get("execution_stage").map(String::as_str) == Some("partial"))
        .collect::<Vec<_>>();
    assert_eq!(partial_strategy_rows.len(), 3);
    let first_decision = partial_strategy_rows[0];
    assert_eq!(first_decision.node_id, aggregate.to_string());
    assert_eq!(first_decision.facts.get("selected_strategy").map(String::as_str), Some("libcudf_groupby"));
    assert_eq!(first_decision.facts.get("source_adjacent_partial").map(String::as_str), Some("false"));
    assert_eq!(first_decision.facts.get("source_chunk_row_group_count").map(String::as_str), Some("1"));
    assert_eq!(first_decision.facts.get("source_chunk_known_row_count").map(String::as_str), Some("40"));
    assert_eq!(
        first_decision.facts.get("encoded_key_partial_rejected_reason").map(String::as_str),
        Some("encoded_key_partial_cost_model_no_prior_cardinality_evidence")
    );
    assert_eq!(
        first_decision.facts.get("encoded_key_partial_selection_reason").map(String::as_str),
        Some("no_prior_cardinality_evidence")
    );
    assert_eq!(
        first_decision.facts.get("encoded_key_partial_cost_model_status").map(String::as_str),
        Some("rejected_by_cost_model")
    );
    assert_eq!(first_decision.facts.get("encode_elapsed_ns").map(String::as_str), Some("0"));
    assert_eq!(first_decision.facts.get("decode_elapsed_ns").map(String::as_str), Some("0"));

    for decision in &partial_strategy_rows[1..] {
        assert_eq!(decision.node_id, aggregate.to_string());
        assert_eq!(decision.facts.get("selected_strategy").map(String::as_str), Some("encoded_key_partial"));
        assert_eq!(decision.facts.get("source_adjacent_partial").map(String::as_str), Some("false"));
        assert_eq!(decision.facts.get("source_chunk_row_group_count").map(String::as_str), Some("1"));
        assert_eq!(decision.facts.get("source_chunk_known_row_count").map(String::as_str), Some("40"));
        assert_eq!(decision.facts.get("encoded_key_partial_rejected_reason").map(String::as_str), Some("none"));
        assert_eq!(decision.facts.get("encoded_key_partial_contract_status").map(String::as_str), Some("available"));
        assert_eq!(
            decision.facts.get("encoded_key_partial_cost_model_status").map(String::as_str),
            Some("selected_low_cardinality")
        );
        assert_eq!(
            decision.facts.get("encoded_key_partial_prior_reduction_ratio_ppm").map(String::as_str),
            Some("75000")
        );
        assert_eq!(
            decision.facts.get("rejected_candidate_strategies").map(String::as_str),
            Some(
                "prepared_sort_groupby:prepared_sort_groupby_requires_source_adjacent_partial;partitioned_merge:partitioned_merge_requires_final_merge_stage"
            )
        );
        assert!(
            decision
                .facts
                .get("encode_elapsed_ns")
                .and_then(|value| value.parse::<u64>().ok())
                .is_some_and(|value| value > 0)
        );
        assert!(
            decision
                .facts
                .get("groupby_elapsed_ns")
                .and_then(|value| value.parse::<u64>().ok())
                .is_some_and(|value| value > 0)
        );
        assert!(
            decision
                .facts
                .get("decode_elapsed_ns")
                .and_then(|value| value.parse::<u64>().ok())
                .is_some_and(|value| value > 0)
        );
    }

    let final_strategy = metrics
        .aggregate_strategy_decisions
        .iter()
        .find(|decision| decision.facts.get("execution_stage").map(String::as_str) == Some("final_merge"))
        .expect("final grouped aggregate merge strategy should be recorded");
    assert_eq!(final_strategy.node_id, aggregate.to_string());
    assert_eq!(final_strategy.facts.get("selected_strategy").map(String::as_str), Some("partitioned_merge"));
    assert_eq!(final_strategy.facts.get("group_key_dtypes").map(String::as_str), Some("Utf8"));
    assert_eq!(final_strategy.facts.get("partial_aggregate_count").map(String::as_str), Some("3"));
    assert_eq!(final_strategy.facts.get("partitioned_merge_contract_status").map(String::as_str), Some("available"));

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(
        grouped_string_sum_count_min_max_rows(batch),
        vec![
            (None, Some(150), 30, Some(5), Some(5)),
            (Some("segment-alpha-long".to_string()), Some(300), 30, Some(10), Some(10)),
            (Some("segment-beta-long".to_string()), Some(60), 60, Some(1), Some(1)),
        ]
    );

    fs::remove_dir_all(dir).unwrap();
}

fn grouped_string_rows(frame: &DataFrame) -> Vec<(Option<String>, Option<i64>, i32)> {
    let batch = frame.to_record_batch().unwrap();
    let keys = string_array(&batch, 0);
    let sums = int64_array(&batch, 1);
    let counts = int32_array(&batch, 2);
    let mut rows = (0..batch.num_rows())
        .map(|row| {
            (
                (!keys.is_null(row)).then(|| keys.value(row).to_string()),
                (!sums.is_null(row)).then(|| sums.value(row)),
                counts.value(row),
            )
        })
        .collect::<Vec<_>>();
    rows.sort();
    rows
}

fn grouped_string_rows_from_batch(batch: &RecordBatch) -> Vec<(Option<String>, Option<i64>, i64)> {
    let keys = string_array(batch, 0);
    let sums = int64_array(batch, 1);
    let counts = int64_array(batch, 2);
    (0..batch.num_rows())
        .map(|row| {
            (
                (!keys.is_null(row)).then(|| keys.value(row).to_string()),
                (!sums.is_null(row)).then(|| sums.value(row)),
                counts.value(row),
            )
        })
        .collect()
}

type GroupedStringSumCountMinMaxRow = (Option<String>, Option<i64>, i64, Option<i64>, Option<i64>);

fn grouped_string_sum_count_min_max_rows(batch: &RecordBatch) -> Vec<GroupedStringSumCountMinMaxRow> {
    let keys = string_array(batch, 0);
    let sums = int64_array(batch, 1);
    let counts = int64_array(batch, 2);
    let mins = int64_array(batch, 3);
    let maxes = int64_array(batch, 4);
    let mut rows = (0..batch.num_rows())
        .map(|row| {
            (
                (!keys.is_null(row)).then(|| keys.value(row).to_string()),
                (!sums.is_null(row)).then(|| sums.value(row)),
                counts.value(row),
                (!mins.is_null(row)).then(|| mins.value(row)),
                (!maxes.is_null(row)).then(|| maxes.value(row)),
            )
        })
        .collect::<Vec<_>>();
    rows.sort();
    rows
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
    let exact_source_read_bytes = u64::try_from(cudf::DataFrame::minimum_parquet_source_read_budget_bytes()).unwrap();
    assert!(metrics.reservation_events.iter().any(|event| {
        event.memory_boundary == "source_chunk_read"
            && event.action == MemoryReservationEventAction::Reserve
            && event.source_read_contract_quality.as_deref() == Some("exact_lower_level_contract")
            && event.requested_bytes == exact_source_read_bytes
    }));

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
    assert_eq!(metrics.partial_aggregate_input_rows, 5);
    assert!(metrics.partial_aggregate_rows >= 3);
    assert!(metrics.partial_aggregate_retained_bytes > 0);
    assert_eq!(metrics.final_merge_attempts, 1);
    assert!(metrics.final_merge_max_fan_in >= 1);
    assert!(
        metrics
            .reservation_events
            .iter()
            .any(|event| { event.memory_boundary == "source_adjacent_partitioned_partial_aggregate_output" })
    );
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
    let aggregate = AggregateSpec::new(
        vec![Expr::column("key")],
        vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("payload"), "count_payload")],
    );
    let sort = SortSpec::new(vec![SortExpr::new(Expr::column("key"), false, false)]);

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let aggregate = builder.add_aggregate(source, aggregate, Arc::clone(&aggregated_schema));
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
    assert_eq!(int64_array(batch, 1).values().as_ref(), &[2, 2, 2]);

    fs::remove_dir_all(dir).unwrap();
}
