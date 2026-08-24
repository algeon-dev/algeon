use std::sync::Arc;

use super::*;

#[test]
fn bounded_native_query_executes_global_final_merge_with_consistent_state_schema() {
    let dir = unique_temp_dir("nexus-query-engine-bounded-global-final-merge-retry");
    let path = dir.join("input.parquet");
    let input_schema = write_final_merge_retry_input_parquet(&path, 1024);
    let repeated_sums = 256;
    let mut fields = (0..repeated_sums)
        .map(|index| Field::new(format!("sum_m00_{index}"), DataType::Int64, true))
        .collect::<Vec<_>>();
    fields.extend([
        Field::new("min_m00", DataType::Int64, true),
        Field::new("max_m00", DataType::Int64, true),
        Field::new("count_m00", DataType::Int64, false),
    ]);
    let aggregated_schema = Arc::new(Schema::new(fields));

    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["m00".to_string()],
    );
    let mut aggr_exprs = (0..repeated_sums)
        .map(|index| AggregateExpr::new(AggregateFunction::Sum, Expr::column("m00"), format!("sum_m00_{index}")))
        .collect::<Vec<_>>();
    aggr_exprs.extend([
        AggregateExpr::new(AggregateFunction::Min, Expr::column("m00"), "min_m00"),
        AggregateExpr::new(AggregateFunction::Max, Expr::column("m00"), "max_m00"),
        AggregateExpr::new(AggregateFunction::Count, Expr::column("m00"), "count_m00"),
    ]);
    let aggregate = AggregateSpec::new(Vec::new(), aggr_exprs);

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let aggregate = builder.add_aggregate(source, aggregate, Arc::clone(&aggregated_schema));
    let plan = builder.finish_with_host_sink(aggregate);

    let context = backend_context_with_memory_control_policy(MemoryControlPolicy::Bounded {
        device_budget_bytes: 4_000_000,
        max_source_chunk_bytes: 1024 * 1024,
        max_row_groups_per_chunk: 1,
        max_retry_attempts: 3,
        min_retry_chunk_row_groups: 1,
    });

    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.source_chunks_planned, 3);
    assert_eq!(metrics.source_chunks_committed, 3);
    assert_eq!(metrics.final_merge_attempts, 1);
    assert_eq!(metrics.final_merge_levels, 1);
    assert_eq!(metrics.retry_count, 0);
    assert_eq!(metrics.retry_reservation_failure_count, 0);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 1);
    assert_eq!(batch.num_columns(), repeated_sums + 3);
    assert!(
        batch.schema().fields().iter().all(|field| !field.name().starts_with("__nexus_native_global_valid_count_"))
    );
    for index in 0..repeated_sums {
        assert_eq!(int64_array(batch, index).value(0), 3072);
    }
    assert_eq!(int64_array(batch, repeated_sums).value(0), 1);
    assert_eq!(int64_array(batch, repeated_sums + 1).value(0), 1);
    assert_eq!(int64_array(batch, repeated_sums + 2).value(0), 3072);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_context_executes_wide_global_final_merge_without_partitioned_path() {
    let dir = unique_temp_dir("nexus-query-engine-raw-bounded-final-merge-retry");
    let path = dir.join("input.parquet");
    let input_schema = write_final_merge_retry_input_parquet(&path, 1024);
    let repeated_sums = 256;
    let aggregated_schema = Arc::new(Schema::new(
        (0..repeated_sums)
            .map(|index| Field::new(format!("sum_m00_{index}"), DataType::Int64, true))
            .collect::<Vec<_>>(),
    ));

    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["m00".to_string()],
    );
    let aggregate = AggregateSpec::new(
        Vec::new(),
        (0..repeated_sums)
            .map(|index| AggregateExpr::new(AggregateFunction::Sum, Expr::column("m00"), format!("sum_m00_{index}")))
            .collect::<Vec<_>>(),
    );

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let aggregate = builder.add_aggregate(source, aggregate, Arc::clone(&aggregated_schema));
    let plan = builder.finish_with_host_sink(aggregate);

    let context = backend_context_with_memory_control_policy(MemoryControlPolicy::Bounded {
        device_budget_bytes: NATIVE_EXECUTION_TEST_QUERY_LIMIT_BYTES,
        max_source_chunk_bytes: 1024 * 1024,
        max_row_groups_per_chunk: 1,
        max_retry_attempts: 3,
        min_retry_chunk_row_groups: 1,
    });

    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.source_chunks_planned, 3);
    assert_eq!(metrics.source_chunks_committed, 3);
    assert_eq!(metrics.final_merge_attempts, 1);
    assert_eq!(metrics.final_merge_levels, 1);
    assert_eq!(metrics.retry_count, 0);
    assert_eq!(metrics.retry_resource_exhausted_count, 0);
    assert_eq!(metrics.retry_reservation_failure_count, 0);
    assert_eq!(metrics.reservation_denied_count, 0);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 1);
    assert_eq!(batch.num_columns(), repeated_sums);
    for index in 0..repeated_sums {
        assert_eq!(int64_array(batch, index).value(0), 3072);
    }

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_context_reports_query_cap_failure_during_final_merge() {
    let dir = unique_temp_dir("nexus-query-engine-raw-bounded-final-merge-retry-exhausted");
    let paths = (0..8).map(|index| dir.join(format!("input{index}.parquet"))).collect::<Vec<_>>();
    let retained_payload_bytes = 64 * 1024;
    let input_schema = write_final_merge_retry_input_parquet_with_payload(&paths[0], 1, retained_payload_bytes);
    for path in &paths[1..] {
        write_final_merge_retry_input_parquet_with_payload(path, 1, retained_payload_bytes);
    }
    let repeated_sums = 128;
    let mut fields = (0..repeated_sums)
        .map(|index| Field::new(format!("sum_m00_{index}"), DataType::Int64, true))
        .collect::<Vec<_>>();
    fields.push(Field::new("avg_m00", DataType::Float64, true));
    fields.push(Field::new("min_payload", DataType::Utf8, true));
    let aggregated_schema = Arc::new(Schema::new(fields));

    let source = LocalParquetSource::from_files(
        paths
            .iter()
            .map(|path| LocalParquetFile::new(path).with_file_size_bytes(fs::metadata(path).unwrap().len()))
            .collect(),
        Arc::clone(&input_schema),
        vec!["m00".to_string(), "payload".to_string()],
    );
    let mut aggr_exprs = (0..repeated_sums)
        .map(|index| AggregateExpr::new(AggregateFunction::Sum, Expr::column("m00"), format!("sum_m00_{index}")))
        .collect::<Vec<_>>();
    aggr_exprs.push(AggregateExpr::new(AggregateFunction::Avg, Expr::column("m00"), "avg_m00"));
    aggr_exprs.push(AggregateExpr::new(AggregateFunction::Min, Expr::column("payload"), "min_payload"));
    let aggregate = AggregateSpec::new(Vec::new(), aggr_exprs);

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let aggregate = builder.add_aggregate(source, aggregate, aggregated_schema);
    let plan = builder.finish_with_host_sink(aggregate);

    let baseline_context = backend_context_with_memory_control_policy(MemoryControlPolicy::Bounded {
        device_budget_bytes: 8 * 1024 * 1024 * 1024,
        max_source_chunk_bytes: 1024 * 1024,
        max_row_groups_per_chunk: 1,
        max_retry_attempts: 1,
        min_retry_chunk_row_groups: 1,
    });
    let baseline = execute_query_plan(&baseline_context, &plan).unwrap();
    let baseline_metrics = baseline.metrics();
    let first_final_merge = baseline_metrics
        .reservation_events
        .iter()
        .position(|event| {
            event.outcome.as_str() == "granted"
                && event.reservation_kind == ReservationKind::FinalMerge
                && event.memory_boundary == "final_aggregate_merge"
        })
        .expect("baseline should reserve final merge work");
    let pre_merge_peak = baseline_metrics.reservation_events[..first_final_merge]
        .iter()
        .map(|event| event.current_reserved_bytes)
        .max()
        .unwrap_or_default();
    let runtime_budget = baseline_metrics
        .reservation_events
        .first()
        .map(|event| event.budget_bytes)
        .expect("baseline should record the runtime reservation budget");
    let hard_floor = |boundary: &str| {
        baseline_metrics
            .budgeted_execution_events
            .iter()
            .filter(|event| event.boundary == boundary)
            .filter_map(|event| event.allocator_high_water_bytes.map(|peak| (event.budget_bytes, peak)))
            .map(|(stage_budget, peak)| {
                runtime_budget
                    .checked_sub(stage_budget)
                    .and_then(|external| external.checked_add(peak))
                    .expect("baseline hard floor should fit inside its runtime budget")
            })
            .max()
            .unwrap_or_else(|| panic!("baseline should report {boundary} allocator high-water bytes"))
    };
    let chunk_operator_hard_floor = hard_floor("chunk_operator");
    let final_merge_hard_floor = hard_floor("final_aggregate_merge");
    let constrained_budget = final_merge_hard_floor.checked_sub(1).expect("final merge hard floor should be nonzero");
    assert!(
        constrained_budget >= pre_merge_peak.max(chunk_operator_hard_floor),
        "fixture must leave a final-merge-only budget window: constrained_budget={constrained_budget}, pre_merge_peak={pre_merge_peak}, chunk_operator_hard_floor={chunk_operator_hard_floor}, final_merge_hard_floor={final_merge_hard_floor}"
    );

    let context = backend_context_with_memory_control_policy(MemoryControlPolicy::Bounded {
        device_budget_bytes: constrained_budget,
        max_source_chunk_bytes: 1024 * 1024,
        max_row_groups_per_chunk: 1,
        max_retry_attempts: 1,
        min_retry_chunk_row_groups: 1,
    });
    let error = match execute_query_plan(&context, &plan) {
        Ok(_) => panic!("bounded execution should fail at the final merge hard cap"),
        Err(error) => error,
    };

    assert_eq!(error.code(), QueryErrorCode::QueryMemoryLimitExceeded);
    assert_eq!(error.operation(), "nexus_query_engine.final_merge.concat_partials");
    assert_eq!(error.fact_value("budget_boundary"), Some("final_aggregate_merge"));
    assert_eq!(error.fact_value("budget_enforcement"), Some("query_domain_member"));
    assert_eq!(error.fact_value("failure_source"), Some("RESERVATION_LIMIT"));
    assert!(error.fact_value("requested_allocation_bytes").is_some());
    assert_eq!(error.fact_value("partial_aggregate_count"), Some("24"));

    fs::remove_dir_all(dir).unwrap();
}
