use super::*;

#[test]
fn bounded_native_query_executes_global_sum_min_max_count_and_final_merge() {
    let dir = unique_temp_dir("nexus-query-engine-bounded-native-global-aggregate-parity");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_row_grouped_input_parquet(&path);

    let aggregated_schema = Arc::new(Schema::new(vec![
        Field::new("sum_m00", DataType::Int64, true),
        Field::new("min_m00", DataType::Int64, true),
        Field::new("max_m01", DataType::Int64, true),
        Field::new("count_payload", DataType::Int64, false),
    ]));

    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["payload".to_string(), "m00".to_string(), "m01".to_string()],
    );
    let filter = Expr::binary(Expr::column("payload"), BinaryOp::GtEq, Expr::literal(LiteralValue::Int64(10)));
    let aggregate = AggregateSpec::new(
        Vec::new(),
        vec![
            AggregateExpr::new(AggregateFunction::Sum, Expr::column("m00"), "sum_m00"),
            AggregateExpr::new(AggregateFunction::Min, Expr::column("m00"), "min_m00"),
            AggregateExpr::new(AggregateFunction::Max, Expr::column("m01"), "max_m01"),
            AggregateExpr::new(AggregateFunction::Count, Expr::column("payload"), "count_payload"),
        ],
    );

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let filter = builder.add_filter(source, filter, Arc::clone(&input_schema));
    let aggregate = builder.add_aggregate(filter, aggregate, aggregated_schema);
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
    assert_eq!(metrics.source_chunks_committed, 1);
    assert_eq!(metrics.partial_aggregate_count, 1);
    assert_eq!(metrics.final_merge_attempts, 0);
    assert_eq!(metrics.boundary.host_to_device_uploads, 0);
    assert_eq!(metrics.boundary.device_to_host_materializations, 1);
    assert_budgeted_success_event_with_contract(
        metrics,
        "nexus_query_engine.global_aggregate.count_output",
        1,
        "chunk_operator",
        Some("libcudf_scalar_to_series_temporary_workspace_preflight_unavailable"),
    );

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 1);
    assert_eq!(int64_array(batch, 0).values().as_ref(), &[1500]);
    assert_eq!(int64_array(batch, 1).values().as_ref(), &[100]);
    assert_eq!(int64_array(batch, 2).values().as_ref(), &[5]);
    assert_eq!(int64_array(batch, 3).values().as_ref(), &[5]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_finalizes_single_chunk_global_sum_min_max_hidden_state() {
    let dir = unique_temp_dir("nexus-query-engine-bounded-native-single-chunk-global-state");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_row_grouped_input_parquet(&path);
    let expected_names = ["sum_m00", "min_m00", "max_m01", "count_payload"];

    let aggregated_schema = Arc::new(Schema::new(vec![
        Field::new("sum_m00", DataType::Int64, true),
        Field::new("min_m00", DataType::Int64, true),
        Field::new("max_m01", DataType::Int64, true),
        Field::new("count_payload", DataType::Int64, false),
    ]));
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["payload".to_string(), "m00".to_string(), "m01".to_string()],
    );
    let aggregate = AggregateSpec::new(
        Vec::new(),
        vec![
            AggregateExpr::new(AggregateFunction::Sum, Expr::column("m00"), "sum_m00"),
            AggregateExpr::new(AggregateFunction::Min, Expr::column("m00"), "min_m00"),
            AggregateExpr::new(AggregateFunction::Max, Expr::column("m01"), "max_m01"),
            AggregateExpr::new(AggregateFunction::Count, Expr::column("payload"), "count_payload"),
        ],
    );

    let build_plan = |host_sink: bool| {
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(source.clone()));
        let filter = Expr::binary(Expr::column("payload"), BinaryOp::GtEq, Expr::literal(LiteralValue::Int64(10)));
        let filter = builder.add_filter(source, filter, Arc::clone(&input_schema));
        let aggregate = builder.add_aggregate(filter, aggregate.clone(), Arc::clone(&aggregated_schema));
        if host_sink { builder.finish_with_host_sink(aggregate) } else { builder.finish_gpu_resident(aggregate) }
    };
    let context = ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(&runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 8 * 1024 * 1024 * 1024,
            max_source_chunk_bytes: 1024 * 1024,
            max_row_groups_per_chunk: 8,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .build()
        .unwrap();

    let host_result = execute_query_plan(&context, &build_plan(true)).unwrap();
    assert_eq!(host_result.metrics().source_chunks_planned, 1);
    assert_eq!(host_result.metrics().partial_aggregate_count, 1);
    assert_eq!(host_result.metrics().final_merge_attempts, 0);
    let QueryOutput::HostArrow(batch) = host_result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_columns(), expected_names.len());
    assert_eq!(batch.schema().fields().iter().map(|field| field.name().as_str()).collect::<Vec<_>>(), expected_names);
    assert!(batch.schema().fields().iter().all(|field| !field.name().starts_with("__cudf_global_valid_count_")));
    assert_eq!(batch.num_rows(), 1);
    assert_eq!(int64_array(batch, 0).values().as_ref(), &[1500]);
    assert_eq!(int64_array(batch, 1).values().as_ref(), &[100]);
    assert_eq!(int64_array(batch, 2).values().as_ref(), &[5]);
    assert_eq!(int64_array(batch, 3).values().as_ref(), &[5]);

    let gpu_result = execute_query_plan(&context, &build_plan(false)).unwrap();
    let QueryOutput::GpuResident(frame) = gpu_result.output() else {
        panic!("gpu sink should return a GPU-resident frame");
    };
    assert_eq!(frame.num_columns(), expected_names.len());
    assert_eq!(frame.frame().column_names(), expected_names);
    assert_eq!(frame.schema().fields().iter().map(|field| field.name().as_str()).collect::<Vec<_>>(), expected_names);
    assert_eq!(frame.device_placement(), context.device_placement());
    assert_eq!(frame.runtime().device().ordinal(), runtime.device().ordinal());
    assert_eq!(gpu_result.metrics().boundary.device_to_host_materializations, 0);

    let mut materialization_metrics = QueryMetrics::default();
    let materialized = materialize_gpu_result_to_host(frame.clone(), &mut materialization_metrics).unwrap();
    assert_eq!(materialization_metrics.boundary.device_to_host_materializations, 1);
    assert_eq!(materialization_metrics.boundary.device_to_host_materialized_rows, 1);
    assert_eq!(
        materialized.schema().fields().iter().map(|field| field.name().as_str()).collect::<Vec<_>>(),
        expected_names
    );
    assert_eq!(int64_array(&materialized, 0).values().as_ref(), &[1500]);

    let output_schema = Arc::new(Schema::new(vec![Field::new("sum_m00", DataType::Int64, true)]));
    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new(
        "q_global_partial",
        Arc::clone(&aggregated_schema),
    )));
    let projection =
        builder.add_projection(source, projection_outputs(vec![Expr::column("sum_m00")], Arc::clone(&output_schema)));
    let consumer_plan = builder.finish_with_host_sink(projection);
    let mut gpu_inputs = GpuResidentInputMap::new();
    gpu_inputs.insert("q_global_partial".to_string(), frame.clone());

    let consumed = execute_query_plan_with_gpu_inputs(&context, &consumer_plan, &gpu_inputs).unwrap();
    assert_eq!(consumed.metrics().boundary.gpu_resident_input_handoffs, 1);
    assert_eq!(consumed.metrics().boundary.host_to_device_uploads, 0);
    assert_eq!(consumed.metrics().boundary.device_to_host_materializations, 1);
    assert_eq!(consumed.metrics().parquet_files, 0);
    let QueryOutput::HostArrow(batch) = consumed.output() else {
        panic!("GPU-resident source consumer should materialize only at the final host sink");
    };
    assert_eq!(batch.schema(), output_schema);
    assert_eq!(int64_array(batch, 0).values().as_ref(), &[1500]);

    fs::remove_dir_all(dir).unwrap();
}
