use super::*;

#[test]
fn encoded_key_partial_contract_decodes_grouped_string_keys_before_merge() {
    let context = backend_context();
    let runtime = Arc::new(context.runtime().unwrap());
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
fn bounded_single_grouped_aggregate_uses_libcudf_without_exact_cardinality_probe() {
    let context = backend_context();
    let row_count = 600_000usize;
    let groups = ["13", "31", "23", "29", "30", "18", "17"];
    let schema = Arc::new(Schema::new(vec![
        Field::new("cntrycode", DataType::Utf8, false),
        Field::new("acctbal", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from_iter_values((0..row_count).map(|row| groups[row % groups.len()]))),
            Arc::new(Int64Array::from_iter_values((0..row_count).map(|_| 1i64))),
        ],
    )
    .unwrap();
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("cntrycode", DataType::Utf8, false),
        Field::new("numcust", DataType::Int64, false),
        Field::new("totacctbal", DataType::Int64, true),
    ]));
    let aggregate_spec = AggregateSpec::new(
        vec![Expr::column("cntrycode")],
        vec![
            AggregateExpr::new(AggregateFunction::Count, Expr::column("acctbal"), "numcust"),
            AggregateExpr::new(AggregateFunction::Sum, Expr::column("acctbal"), "totacctbal"),
        ],
    );

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("grouped", Arc::clone(&schema))));
    let aggregate = builder.add_aggregate(source, aggregate_spec, aggregate_schema);
    let plan = builder.finish_with_host_sink(aggregate);

    let admitted = context.admit_plan(&plan).unwrap();
    let runtime = Arc::new(admitted.runtime().unwrap());
    let mut inputs = GpuResidentInputMap::new();
    inputs.insert(
        "grouped".to_owned(),
        GpuResultHandle::new(
            &admitted,
            GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &batch).unwrap()),
            Arc::clone(&schema),
        )
        .unwrap(),
    );
    let result = execute_query_plan_with_gpu_inputs(&admitted, &plan, &inputs).unwrap();
    let metrics = result.metrics();
    let single_strategy = metrics
        .aggregate_strategy_decisions
        .iter()
        .find(|decision| decision.facts.get("execution_stage").map(String::as_str) == Some("single"))
        .expect("single aggregate should record a strategy decision");
    assert_eq!(single_strategy.node_id, aggregate.to_string());
    assert_eq!(single_strategy.facts.get("selected_strategy").map(String::as_str), Some("libcudf_groupby"));
    assert_eq!(
        single_strategy.facts.get("exact_cardinality_probe_status").map(String::as_str),
        Some("disabled_non_reusable_probe")
    );
    assert_ne!(
        single_strategy.facts.get("prepared_sort_groupby_cost_model_status").map(String::as_str),
        Some("selected_exact_cardinality_low_cardinality")
    );

    let probe_events =
        budgeted_events(metrics, "nexus_query_engine.aggregate.exact_single_key_cardinality_probe.nunique");
    assert!(probe_events.is_empty(), "hot path must not launch throwaway exact-cardinality probes");
    let prepare_events = budgeted_events(metrics, "nexus_query_engine.aggregate.prepare_single.prepared_sort_groupby");
    assert!(prepare_events.is_empty(), "unseen single aggregates must not select prepared sort-groupby");

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), groups.len());
    let output_keys = batch.column(0).as_any().downcast_ref::<StringArray>().unwrap();
    let output_counts = batch.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
    let output_sums = batch.column(2).as_any().downcast_ref::<Int64Array>().unwrap();
    let mut actual = (0..batch.num_rows())
        .map(|row| (output_keys.value(row).to_string(), output_counts.value(row), output_sums.value(row)))
        .collect::<Vec<_>>();
    actual.sort_by(|left, right| left.0.cmp(&right.0));
    let mut expected_counts = std::collections::BTreeMap::<String, i64>::new();
    for row in 0..row_count {
        *expected_counts.entry(groups[row % groups.len()].to_string()).or_insert(0) += 1;
    }
    let expected = expected_counts.into_iter().map(|(key, count)| (key, count, count)).collect::<Vec<_>>();
    assert_eq!(actual, expected);
    admitted.finish();
}

#[test]
fn bounded_medium_single_string_avg_uses_libcudf_without_exact_cardinality_probe() {
    let context = backend_context();
    let row_count = 204_000usize;
    let groups =
        ["Books", "Children", "Electronics", "Home", "Jewelry", "Men", "Music", "Shoes", "Sports", "Tools", "Women"];
    let schema = Arc::new(Schema::new(vec![
        Field::new("category", DataType::Utf8, false),
        Field::new("price", DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from_iter_values((0..row_count).map(|row| groups[row % groups.len()]))),
            Arc::new(Float64Array::from_iter_values((0..row_count).map(|row| (row % groups.len()) as f64 + 1.0))),
        ],
    )
    .unwrap();
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("category", DataType::Utf8, false),
        Field::new("avg_price", DataType::Float64, true),
    ]));
    let aggregate_spec = AggregateSpec::new(
        vec![Expr::column("category")],
        vec![AggregateExpr::new(AggregateFunction::Avg, Expr::column("price"), "avg_price")],
    );

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("grouped", Arc::clone(&schema))));
    let aggregate = builder.add_aggregate(source, aggregate_spec, Arc::clone(&aggregate_schema));
    let plan = builder.finish_with_host_sink(aggregate);

    let admitted = context.admit_plan(&plan).unwrap();
    let runtime = Arc::new(admitted.runtime().unwrap());
    let mut inputs = GpuResidentInputMap::new();
    inputs.insert(
        "grouped".to_owned(),
        GpuResultHandle::new(
            &admitted,
            GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &batch).unwrap()),
            Arc::clone(&schema),
        )
        .unwrap(),
    );
    let result = execute_query_plan_with_gpu_inputs(&admitted, &plan, &inputs).unwrap();
    let metrics = result.metrics();
    let single_strategy = metrics
        .aggregate_strategy_decisions
        .iter()
        .find(|decision| decision.facts.get("execution_stage").map(String::as_str) == Some("single"))
        .expect("single aggregate should record a strategy decision");
    assert_eq!(single_strategy.node_id, aggregate.to_string());
    assert_eq!(single_strategy.facts.get("selected_strategy").map(String::as_str), Some("libcudf_groupby"));
    assert_eq!(
        single_strategy.facts.get("exact_cardinality_probe_status").map(String::as_str),
        Some("disabled_non_reusable_probe")
    );

    let probe_events =
        budgeted_events(metrics, "nexus_query_engine.aggregate.exact_single_key_cardinality_probe.nunique");
    assert!(probe_events.is_empty(), "hot path must not launch throwaway exact-cardinality probes");
    let prepare_events = budgeted_events(metrics, "nexus_query_engine.aggregate.prepare_single.prepared_sort_groupby");
    assert!(prepare_events.is_empty(), "unseen single aggregates must not select prepared sort-groupby");
    let avg_events = budgeted_events(metrics, "nexus_query_engine.aggregate.finalize_avg");
    assert!(!avg_events.is_empty(), "AVG should record AVG finalization");

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.schema(), aggregate_schema);
    assert_eq!(batch.num_rows(), groups.len());
    let output_keys = batch.column(0).as_any().downcast_ref::<StringArray>().unwrap();
    let output_avg = batch.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
    let mut actual = (0..batch.num_rows())
        .map(|row| (output_keys.value(row).to_string(), output_avg.value(row)))
        .collect::<Vec<_>>();
    actual.sort_by(|left, right| left.0.cmp(&right.0));
    let mut expected =
        groups.iter().enumerate().map(|(index, group)| ((*group).to_string(), index as f64 + 1.0)).collect::<Vec<_>>();
    expected.sort_by(|left, right| left.0.cmp(&right.0));
    assert_eq!(actual, expected);
    admitted.finish();
}

#[test]
fn bounded_native_query_executes_chunked_grouped_sum_and_final_merge() {
    let dir = unique_temp_dir("nexus-query-engine-bounded-native-execution");
    let path = dir.join("input.parquet");
    let device_budget_bytes = 8 * 1024 * 1024 * 1024;
    let context = backend_context_with_grant(device_budget_bytes);
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
        vec![
            LocalParquetFile::new(&path)
                .with_file_size_bytes(fs::metadata(&path).unwrap().len())
                .with_row_group_bytes(vec![2 * 1024 * 1024; 3])
                .with_projected_row_group_bytes(vec![2 * 1024 * 1024; 3], vec![2 * 1024 * 1024; 3])
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

    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.memory_control_policy.as_str(), "bounded");
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
                "encoded_key_partial:encoded_key_partial_requires_string_group_key;prepared_sort_groupby:prepared_sort_groupby_cost_model_small_input;partitioned_merge:partitioned_merge_requires_final_merge_stage"
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
    let source_events = budgeted_events(metrics, "nexus_query_engine.local_parquet_chunk.read");
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
fn bounded_native_query_batches_source_adjacent_grouped_aggregate_files() {
    let dir = unique_temp_dir("nexus-query-engine-bounded-source-adjacent-batched-files");
    let paths = [dir.join("input-0.parquet"), dir.join("input-1.parquet"), dir.join("input-2.parquet")];
    let context = backend_context();
    let runtime = Arc::new(context.runtime().unwrap());
    let input_schema = write_input_parquet(&runtime, &paths[0]);
    write_input_parquet(&runtime, &paths[1]);
    write_input_parquet(&runtime, &paths[2]);

    let projected_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("x00", DataType::Int64, false),
    ]));
    let aggregated_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("sum_x00", DataType::Int64, true),
    ]));
    let files = paths
        .iter()
        .map(|path| {
            LocalParquetFile::new(path)
                .with_file_size_bytes(fs::metadata(path).unwrap().len())
                .with_row_group_bytes(vec![64 * 1024])
                .with_projected_row_group_bytes(vec![64 * 1024], vec![64 * 1024])
                .with_row_group_rows(vec![4])
        })
        .collect::<Vec<_>>();
    let source = LocalParquetSource::from_files(
        files,
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
    let plan = builder.finish_with_host_sink(sort);

    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.source_chunks_planned, 3);
    assert_eq!(metrics.source_chunks_attempted, 3);
    assert_eq!(metrics.source_chunks_committed, 3);
    assert_eq!(metrics.row_groups_planned, 3);
    assert_eq!(metrics.row_groups_committed, 3);
    assert_eq!(metrics.source_read_prefetches_started, 0);
    assert_eq!(metrics.source_read_runtime_rows.len(), 1);
    let source_read_row = &metrics.source_read_runtime_rows[0];
    assert_eq!(source_read_row.read_kind, "chunk_batch");
    assert_eq!(source_read_row.chunk_count, 3);
    assert_eq!(source_read_row.file_count, 3);
    assert_eq!(source_read_row.read_columns_total, 12);
    assert!(metrics.partial_aggregate_count > 0);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[3, 1, 2]);
    assert_eq!(int64_array(batch, 1).values().as_ref(), &[1212, 909, 606]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_routes_many_string_morsels_away_from_source_adjacent_partitions() {
    let dir = unique_temp_dir("nexus-query-engine-bounded-source-adjacent-many-string-morsels");
    let path = dir.join("input.parquet");
    let context = backend_context();
    let row_group_count = 65;
    let input_schema = write_many_row_group_q01_like_parquet(&path, row_group_count);

    let projected_schema = Arc::new(Schema::new(vec![
        Field::new("l_returnflag", DataType::Utf8, true),
        Field::new("l_linestatus", DataType::Utf8, true),
        Field::new("value", DataType::Int64, true),
    ]));
    let aggregated_schema = Arc::new(Schema::new(vec![
        Field::new("l_returnflag", DataType::Utf8, true),
        Field::new("l_linestatus", DataType::Utf8, true),
        Field::new("sum_value", DataType::Int64, true),
    ]));

    let source = LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new(&path)
                .with_file_size_bytes(fs::metadata(&path).unwrap().len())
                .with_row_group_bytes(vec![1024 * 1024; row_group_count])
                .with_projected_row_group_bytes(vec![1024 * 1024; row_group_count], vec![1024 * 1024; row_group_count])
                .with_row_group_rows(vec![1; row_group_count]),
        ],
        Arc::clone(&input_schema),
        vec!["flag".to_string(), "status".to_string(), "value".to_string()],
    );
    let projection = vec![
        Expr::column("flag").alias("l_returnflag"),
        Expr::column("status").alias("l_linestatus"),
        Expr::column("value"),
    ];
    let aggregate = AggregateSpec::new(
        vec![Expr::column("l_returnflag"), Expr::column("l_linestatus")],
        vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("value"), "sum_value")],
    );

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let projection = builder.add_projection(source, projection_outputs(projection, projected_schema));
    let aggregate = builder.add_aggregate(projection, aggregate, aggregated_schema);
    let plan = builder.finish_with_host_sink(aggregate);

    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.source_chunks_planned, row_group_count as u64);
    assert_eq!(metrics.source_chunks_committed, row_group_count as u64);
    assert_eq!(metrics.row_groups_planned, row_group_count as u64);
    assert_eq!(metrics.row_groups_committed, row_group_count as u64);

    let partial_strategy_rows = metrics
        .aggregate_strategy_decisions
        .iter()
        .filter(|decision| decision.facts.get("execution_stage").map(String::as_str) == Some("partial"))
        .collect::<Vec<_>>();
    assert!(!partial_strategy_rows.is_empty());
    // Variable-width group keys take the group-key hash partition path even when
    // source chunk count exceeds the old 64-morsel gate; admission is live-set based.
    assert!(partial_strategy_rows.iter().all(|decision| {
        decision.facts.get("source_adjacent_partition_strategy").map(String::as_str) == Some("group_key")
    }));
    assert!(
        metrics
            .reservation_events
            .iter()
            .any(|event| event.memory_boundary == "source_adjacent_aggregate_hash_partition"),
        "many string row-group morsels should stream through group-key hash partition under live working-set admission"
    );

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert!(batch.num_rows() <= 4);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_streams_source_adjacent_grouped_chunks_when_partition_retention_exceeds_budget() {
    let dir = unique_temp_dir("nexus-query-engine-bounded-source-adjacent-retention-budget");
    let path = dir.join("input.parquet");
    let device_budget_bytes = 8 * 1024 * 1024 * 1024;
    let context = backend_context_with_grant(device_budget_bytes);
    let input_schema = write_row_grouped_input_parquet(&path);
    let projected_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("x00", DataType::Int64, false),
    ]));
    let aggregated_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("sum_x00", DataType::Int64, true),
    ]));

    let four_gib = 4 * 1024 * 1024 * 1024;
    let source = LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new(&path)
                .with_file_size_bytes(fs::metadata(&path).unwrap().len())
                .with_row_group_bytes(vec![four_gib; 3])
                .with_projected_row_group_bytes(vec![four_gib; 3], vec![four_gib; 3])
                .with_row_group_rows(vec![2, 2, 2]),
        ],
        Arc::clone(&input_schema),
        vec!["key".to_string(), "m00".to_string(), "m01".to_string()],
    );
    let projection =
        vec![Expr::column("key"), Expr::binary(Expr::column("m00"), BinaryOp::Add, Expr::column("m01")).alias("x00")];
    let aggregate = AggregateSpec::new(
        vec![Expr::column("key")],
        vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("x00"), "sum_x00")],
    );

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let projection = builder.add_projection(source, projection_outputs(projection, projected_schema));
    let aggregate = builder.add_aggregate(projection, aggregate, aggregated_schema);
    let plan = builder.finish_with_host_sink(aggregate);

    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.source_chunks_planned, 3);
    assert_eq!(metrics.partial_aggregate_count, 3);
    assert!(
        metrics
            .reservation_events
            .iter()
            .all(|event| event.memory_boundary != "source_adjacent_aggregate_hash_partition"),
        "partitioned source-adjacent aggregation must not retain raw partitions when estimates exceed the budget"
    );

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 3);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_executes_source_adjacent_grouped_sum_with_empty_partitions() {
    let dir = unique_temp_dir("nexus-query-engine-bounded-source-adjacent-empty-partitions");
    let path = dir.join("input.parquet");
    let context = backend_context_with_memory_and_scheduler_policy(
        MemoryControlPolicy::Bounded {
            device_budget_bytes: 8 * 1024 * 1024 * 1024,
            max_source_chunk_bytes: 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        },
        SourceSchedulerPolicy::new(3),
    );
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
    let dir = unique_temp_dir("nexus-query-engine-bounded-source-adjacent-group-key-partitions");
    let path = dir.join("input.parquet");
    let context = backend_context();
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
        vec![
            LocalParquetFile::new(&path)
                .with_file_size_bytes(fs::metadata(&path).unwrap().len())
                .with_row_group_bytes(vec![2 * 1024 * 1024; 3])
                .with_projected_row_group_bytes(vec![2 * 1024 * 1024; 3], vec![2 * 1024 * 1024; 3])
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
            AggregateExpr::new(AggregateFunction::Min, Expr::column("value"), "min_value"),
            AggregateExpr::new(AggregateFunction::Max, Expr::column("value"), "max_value"),
        ],
    );

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let projection = builder.add_projection(source, projection_outputs(projection, projected_schema));
    let aggregate = builder.add_aggregate(projection, aggregate, Arc::clone(&aggregated_schema));
    let plan = builder.finish_with_host_sink(aggregate);

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
    // Source-adjacent envelope alone must not force prepared sort on the first partial.
    assert_ne!(
        first_partial.facts.get("prepared_sort_groupby_cost_model_status").map(String::as_str),
        Some("selected_source_adjacent_execution")
    );
    assert_ne!(
        first_partial
            .facts
            .get("selected_strategy")
            .map(String::as_str)
            .zip(first_partial.facts.get("prepared_sort_groupby_selection_reason").map(String::as_str),),
        Some(("prepared_sort_groupby", "source_adjacent_partial_prepared_execution"))
    );
    // Allow subsequent identical partitions to reuse completed prior evidence for prepared/encoded.
    assert!(partial_strategy_rows.iter().all(|decision| {
        matches!(
            decision.facts.get("selected_strategy").map(String::as_str),
            Some("libcudf_groupby") | Some("encoded_key_partial") | Some("prepared_sort_groupby")
        )
    }));

    let final_strategy = metrics
        .aggregate_strategy_decisions
        .iter()
        .find(|decision| decision.facts.get("execution_stage").map(String::as_str) == Some("final_merge"))
        .expect("group-key source-adjacent aggregate should record final merge strategy");
    assert_eq!(final_strategy.node_id, aggregate.to_string());
    assert_eq!(final_strategy.facts.get("selected_strategy").map(String::as_str), Some("partitioned_merge"));
    assert_eq!(final_strategy.facts.get("partitioned_merge_rehash_skipped").map(String::as_str), Some("true"));
    assert_eq!(
        final_strategy.facts.get("partitioned_merge_aligned_input").map(String::as_str),
        Some("group_key_source_adjacent")
    );

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
fn bounded_native_query_keeps_large_string_source_adjacent_inputs_on_group_key_partitions() {
    let dir = unique_temp_dir("nexus-query-engine-bounded-source-adjacent-large-string-group-key");
    let path = dir.join("input.parquet");
    let context = backend_context();
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

    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert!(metrics.partial_aggregate_count > 0);
    let partial_strategy = metrics
        .aggregate_strategy_decisions
        .iter()
        .find(|decision| decision.facts.get("execution_stage").map(String::as_str) == Some("partial"))
        .expect("row-balanced partial strategy row");
    assert_eq!(partial_strategy.facts.get("source_adjacent_partition_strategy").map(String::as_str), Some("group_key"));
    assert!(
        metrics.aggregate_strategy_decisions.iter().all(|decision| {
            decision.facts.get("source_adjacent_partition_strategy").map(String::as_str) != Some("group_key")
                || decision
                    .facts
                    .get("source_adjacent_max_partition_fan_in")
                    .and_then(|fan_in| fan_in.parse::<u64>().ok())
                    .is_some_and(|fan_in| fan_in <= 64)
                || decision.facts.get("selected_strategy").map(String::as_str) != Some("prepared_sort_groupby")
        }),
        "high-fan-in group-key source-adjacent string aggregates must not use prepared sort groupby"
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
fn bounded_native_query_prunes_partitioned_final_merge_outputs_with_topk() {
    let dir = unique_temp_dir("nexus-query-engine-bounded-partitioned-final-merge-topk");
    let path = dir.join("input.parquet");
    let context = backend_context();
    let input_schema = write_high_cardinality_url_count_parquet(&path);

    let projected_schema = Arc::new(Schema::new(vec![Field::new("URL", DataType::Utf8, false)]));
    let aggregated_schema =
        Arc::new(Schema::new(vec![Field::new("URL", DataType::Utf8, false), Field::new("c", DataType::Int64, false)]));

    let source = LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new(&path)
                .with_file_size_bytes(fs::metadata(&path).unwrap().len())
                .with_row_group_bytes(vec![2 * 1024 * 1024; 13])
                .with_projected_row_group_bytes(vec![2 * 1024 * 1024; 13], vec![2 * 1024 * 1024; 13])
                .with_row_group_rows([vec![64; 12], vec![52]].concat()),
        ],
        Arc::clone(&input_schema),
        vec!["URL".to_string()],
    );
    let projection = vec![Expr::column("URL")];
    let aggregate = AggregateSpec::new(
        vec![Expr::column("URL")],
        vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("URL"), "c")],
    );
    let sort = SortSpec::new(vec![
        SortExpr::new(Expr::column("c"), true, false),
        SortExpr::new(Expr::column("URL"), false, false),
    ]);

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let projection = builder.add_projection(source, projection_outputs(projection, projected_schema));
    let aggregate = builder.add_aggregate(projection, aggregate, Arc::clone(&aggregated_schema));
    let aggregate_projection = builder.add_projection(
        aggregate,
        projection_outputs(vec![Expr::column("URL"), Expr::column("c")], Arc::clone(&aggregated_schema)),
    );
    let sort = builder.add_sort(aggregate_projection, sort, Arc::clone(&aggregated_schema));
    let limit = builder.add_limit(sort, LimitSpec::fetch(2), aggregated_schema);
    let plan = builder.finish_with_host_sink(limit);

    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    let final_strategy = metrics
        .aggregate_strategy_decisions
        .iter()
        .find(|decision| decision.facts.get("execution_stage").map(String::as_str) == Some("final_merge"))
        .expect("partitioned final merge should record strategy facts");
    assert_eq!(final_strategy.node_id, aggregate.to_string());
    assert_eq!(final_strategy.facts.get("selected_strategy").map(String::as_str), Some("partitioned_merge"));
    assert_eq!(final_strategy.facts.get("partitioned_merge_topk_pruned").map(String::as_str), Some("true"));
    assert_eq!(final_strategy.facts.get("partitioned_merge_topk_keep_rows").map(String::as_str), Some("2"));
    let topk_input_rows = final_strategy
        .facts
        .get("partitioned_merge_topk_input_rows")
        .and_then(|value| value.parse::<u64>().ok())
        .expect("top-k input row fact");
    let topk_output_rows = final_strategy
        .facts
        .get("partitioned_merge_topk_output_rows")
        .and_then(|value| value.parse::<u64>().ok())
        .expect("top-k output row fact");
    assert!(
        topk_output_rows < topk_input_rows,
        "partition-local TopK should reduce high-cardinality final aggregate output before concat"
    );
    assert!(metrics.reservation_events.iter().any(|event| {
        event.memory_boundary == "partitioned_final_aggregate_topk_input"
            && event.action == MemoryReservationEventAction::Reserve
    }));

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 2);
    assert_eq!(string_array(batch, 0).iter().collect::<Vec<_>>(), vec![Some("url-039"), Some("url-038")]);
    assert_eq!(int64_array(batch, 1).values().as_ref(), &[40, 39]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_keeps_small_nullable_string_partials_on_libcudf_before_final_merge() {
    let dir = unique_temp_dir("nexus-query-engine-bounded-encoded-string-aggregate");
    let path = dir.join("input.parquet");
    let context = backend_context();
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
    for decision in &partial_strategy_rows {
        assert_eq!(decision.node_id, aggregate.to_string());
        assert_eq!(decision.facts.get("selected_strategy").map(String::as_str), Some("libcudf_groupby"));
        assert_eq!(decision.facts.get("source_adjacent_partial").map(String::as_str), Some("false"));
        assert_eq!(decision.facts.get("source_chunk_row_group_count").map(String::as_str), Some("1"));
        assert_eq!(decision.facts.get("source_chunk_known_row_count").map(String::as_str), Some("40"));
        assert_eq!(
            decision.facts.get("encoded_key_partial_rejected_reason").map(String::as_str),
            Some("encoded_key_partial_cost_model_small_input")
        );
        assert_eq!(
            decision.facts.get("encoded_key_partial_selection_reason").map(String::as_str),
            Some("input_rows_below_encoded_key_partial_minimum")
        );
        assert_eq!(
            decision.facts.get("encoded_key_partial_cost_model_status").map(String::as_str),
            Some("rejected_by_cost_model")
        );
        assert_eq!(decision.facts.get("encode_elapsed_ns").map(String::as_str), Some("0"));
        assert!(
            decision
                .facts
                .get("groupby_elapsed_ns")
                .and_then(|value| value.parse::<u64>().ok())
                .is_some_and(|value| value > 0)
        );
        assert_eq!(decision.facts.get("decode_elapsed_ns").map(String::as_str), Some("0"));
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

fn write_many_row_group_q01_like_parquet(path: &Path, row_group_count: usize) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("flag", DataType::Utf8, true),
        Field::new("status", DataType::Utf8, true),
        Field::new("value", DataType::Int64, true),
    ]));
    let flags = (0..row_group_count).map(|row| if row % 2 == 0 { Some("A") } else { Some("R") }).collect::<Vec<_>>();
    let statuses = (0..row_group_count).map(|row| if row % 3 == 0 { Some("F") } else { Some("O") }).collect::<Vec<_>>();
    let values = (0..row_group_count).map(|row| Some(i64::try_from(row + 1).unwrap())).collect::<Vec<_>>();
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(flags)),
            Arc::new(StringArray::from(statuses)),
            Arc::new(Int64Array::from(values)),
        ],
    )
    .unwrap();
    write_record_batch_with_row_group_rows(path, Arc::clone(&schema), batch, 1);
    schema
}

fn write_high_cardinality_url_count_parquet(path: &Path) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![Field::new("URL", DataType::Utf8, false)]));
    let mut urls = Vec::new();
    for group in 0..40 {
        urls.extend(std::iter::repeat_n(format!("url-{group:03}"), group + 1));
    }
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(StringArray::from(urls))]).unwrap();
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(64)).build();
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    schema
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
fn bounded_native_query_executes_min_max_count_and_final_merge() {
    let dir = unique_temp_dir("nexus-query-engine-bounded-native-aggregate-parity");
    let path = dir.join("input.parquet");
    let context = backend_context();
    let input_schema = write_row_grouped_input_parquet(&path);

    let aggregated_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("min_m00", DataType::Int64, true),
        Field::new("max_m01", DataType::Int64, true),
        Field::new("count_payload", DataType::Int64, false),
    ]));

    let source = LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new(&path)
                .with_file_size_bytes(fs::metadata(&path).unwrap().len())
                .with_row_group_bytes(vec![2 * 1024 * 1024; 3])
                .with_projected_row_group_bytes(vec![2 * 1024 * 1024; 3], vec![2 * 1024 * 1024; 3])
                .with_row_group_rows(vec![2, 2, 2]),
        ],
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
    let dir = unique_temp_dir("nexus-query-engine-bounded-native-single-partial-count");
    let path = dir.join("input.parquet");
    let context = backend_context_with_memory_control_policy(MemoryControlPolicy::Bounded {
        device_budget_bytes: 8 * 1024 * 1024 * 1024,
        max_source_chunk_bytes: 8 * 1024 * 1024 * 1024,
        max_row_groups_per_chunk: 16,
        max_retry_attempts: 1,
        min_retry_chunk_row_groups: 1,
    });
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

#[test]
fn bounded_native_query_coalesces_many_source_adjacent_grouped_row_groups_by_budget() {
    let dir = unique_temp_dir("nexus-query-engine-bounded-native-many-source-adjacent-chunks");
    let path = dir.join("input.parquet");
    let row_group_count = 65;
    let input_schema = write_many_row_group_q01_like_parquet(&path, row_group_count);

    let projected_schema = Arc::new(Schema::new(vec![
        Field::new("l_returnflag", DataType::Utf8, true),
        Field::new("l_linestatus", DataType::Utf8, true),
        Field::new("value", DataType::Int64, true),
    ]));
    let aggregated_schema = Arc::new(Schema::new(vec![
        Field::new("l_returnflag", DataType::Utf8, true),
        Field::new("l_linestatus", DataType::Utf8, true),
        Field::new("sum_value", DataType::Int64, true),
    ]));

    let row_group_bytes = 64 * 1024 * 1024;
    let source = LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new(&path)
                .with_file_size_bytes(fs::metadata(&path).unwrap().len())
                .with_row_group_bytes(vec![row_group_bytes; row_group_count])
                .with_projected_row_group_bytes(
                    vec![row_group_bytes; row_group_count],
                    vec![row_group_bytes; row_group_count],
                )
                .with_row_group_rows(vec![1; row_group_count]),
        ],
        Arc::clone(&input_schema),
        vec!["flag".to_string(), "status".to_string(), "value".to_string()],
    );
    let projection = vec![
        Expr::column("flag").alias("l_returnflag"),
        Expr::column("status").alias("l_linestatus"),
        Expr::column("value"),
    ];
    let aggregate = AggregateSpec::new(
        vec![Expr::column("l_returnflag"), Expr::column("l_linestatus")],
        vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("value"), "sum_value")],
    );

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let projection = builder.add_projection(source, projection_outputs(projection, projected_schema));
    let aggregate = builder.add_aggregate(projection, aggregate, aggregated_schema);
    let plan = builder.finish_with_host_sink(aggregate);

    let device_budget_bytes = 8 * 1024 * 1024 * 1024;
    let context = backend_context_with_memory_control_policy(MemoryControlPolicy::Bounded {
        device_budget_bytes,
        max_source_chunk_bytes: device_budget_bytes,
        max_row_groups_per_chunk: 1024,
        max_retry_attempts: 1,
        min_retry_chunk_row_groups: 1,
    });
    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.source_chunks_planned, 11);
    assert_eq!(metrics.source_chunks_committed, 11);
    assert_eq!(metrics.row_groups_planned, row_group_count as u64);
    assert_eq!(metrics.row_groups_committed, row_group_count as u64);
    assert!(metrics.partial_aggregate_count > 0);
    assert!(metrics.partial_aggregate_count < row_group_count as u64);
    assert!(metrics.final_merge_attempts > 0);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert!(batch.num_rows() <= 4);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_uses_configured_source_cap_when_budget_model_allows_it() {
    let dir = unique_temp_dir("nexus-query-engine-bounded-aggregate-source-refinement");
    let path = dir.join("input.parquet");
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

    let context = backend_context_with_memory_control_policy(MemoryControlPolicy::Bounded {
        device_budget_bytes: 8 * 1024 * 1024 * 1024,
        max_source_chunk_bytes: 1600,
        max_row_groups_per_chunk: 8,
        max_retry_attempts: 1,
        min_retry_chunk_row_groups: 1,
    });
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
