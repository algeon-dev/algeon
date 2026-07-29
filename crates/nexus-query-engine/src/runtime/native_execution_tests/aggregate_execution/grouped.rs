use super::*;

#[test]
fn bounded_native_query_coalesces_many_source_adjacent_grouped_row_groups_by_budget() {
    let dir = unique_temp_dir("nexus-query-engine-bounded-native-many-source-adjacent-chunks");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
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
    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes,
            max_source_chunk_bytes: device_budget_bytes,
            max_row_groups_per_chunk: 1024,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .build()
        .unwrap();
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
