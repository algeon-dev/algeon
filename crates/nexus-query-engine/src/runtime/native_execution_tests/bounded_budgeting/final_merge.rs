use super::*;

#[test]
fn raw_bounded_context_executes_wide_global_final_merge_without_partitioned_path() {
    let dir = unique_temp_dir("nexus-query-engine-raw-bounded-final-merge-retry");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
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

    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 300_000,
            max_source_chunk_bytes: 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 3,
            min_retry_chunk_row_groups: 1,
        })
        .build()
        .unwrap();
    assert_eq!(
        context.memory_control_policy(),
        MemoryControlPolicy::Bounded {
            device_budget_bytes: 300_000,
            max_source_chunk_bytes: 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 3,
            min_retry_chunk_row_groups: 1,
        }
    );

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
fn raw_bounded_context_reports_retry_facts_when_final_merge_retry_exhausts() {
    let dir = unique_temp_dir("nexus-query-engine-raw-bounded-final-merge-retry-exhausted");
    let path = dir.join("input.parquet");
    let second_path = dir.join("input2.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_final_merge_retry_input_parquet(&path, 1024);
    write_final_merge_retry_input_parquet(&second_path, 1024);
    let repeated_sums = 128;
    let aggregated_schema = Arc::new(Schema::new(
        (0..repeated_sums)
            .map(|index| Field::new(format!("sum_m00_{index}"), DataType::Int64, true))
            .collect::<Vec<_>>(),
    ));

    let source = LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len()),
            LocalParquetFile::new(&second_path).with_file_size_bytes(fs::metadata(&second_path).unwrap().len()),
        ],
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
    let aggregate = builder.add_aggregate(source, aggregate, aggregated_schema);
    let plan = builder.finish_with_host_sink(aggregate);

    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 140_000,
            max_source_chunk_bytes: 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .build()
        .unwrap();
    assert_eq!(
        context.memory_control_policy(),
        MemoryControlPolicy::Bounded {
            device_budget_bytes: 140_000,
            max_source_chunk_bytes: 1024 * 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        }
    );

    let error = match execute_query_plan(&context, &plan) {
        Ok(_) => panic!("bounded execution should fail after exhausting final merge retry fan-in"),
        Err(error) => error,
    };

    assert!(matches!(error.code(), QueryErrorCode::MemoryReservationFailed | QueryErrorCode::CudfResourceExhausted));
    assert_eq!(error.fact_value("memory_boundary"), Some("final_aggregate_merge"));
    assert_eq!(error.fact_value("stage"), Some("final_merge"));
    assert_eq!(error.fact_value("attempt"), Some("1"));
    assert_eq!(error.fact_value("max_attempts"), Some("1"));
    assert_eq!(error.fact_value("retry_action"), Some("retry_attempts_exhausted"));
    assert_eq!(error.fact_value("resume_boundary"), Some("final_merge"));
    assert_eq!(error.fact_value("committed_chunks"), Some("6"));
    assert_eq!(error.fact_value("committed_partials"), Some("6"));
    assert_eq!(error.fact_value("merge_level"), Some("1"));
    assert_eq!(error.fact_value("merge_fan_in"), Some("6"));
    if error.code() == QueryErrorCode::MemoryReservationFailed {
        assert_eq!(error.fact_value("reservation_kind"), Some("final_merge"));
    } else {
        assert_eq!(error.fact_value("budget_boundary"), Some("final_aggregate_merge"));
        assert_eq!(error.fact_value("budget_enforcement"), Some("stream_bound_reservation"));
    }

    fs::remove_dir_all(dir).unwrap();
}
