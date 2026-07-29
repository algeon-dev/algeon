use super::*;

#[test]
fn bounded_native_query_reports_reservation_failure_before_host_materialization() {
    let dir = unique_temp_dir("nexus-query-engine-bounded-materialization-reservation-failure");
    let path = dir.join("input.parquet");
    let input_schema = write_single_row_group_final_sort_input_parquet(&path, 4096);
    let aggregated_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("sum_m00", DataType::Int64, true),
    ]));
    let materialized_schema = wide_final_sort_schema(32);

    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["key".to_string(), "m00".to_string()],
    );
    let aggregate = AggregateSpec::new(
        vec![Expr::column("key")],
        vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("m00"), "sum_m00")],
    );

    let mut builder = QueryPlanBuilder::new();
    let source_id = builder.add_source(SourceNode::LocalParquet(source.clone()));
    let one_chunk_plan =
        plan_local_parquet_source_chunks(source_id, &source, SourceChunkPlanningPolicy::new(None, 1)).unwrap();
    assert_eq!(one_chunk_plan.chunk_count(), 1);
    let aggregate = builder.add_aggregate(source_id, aggregate, Arc::clone(&aggregated_schema));
    let projection = builder.add_projection(
        aggregate,
        projection_outputs(wide_final_sort_projection(32), Arc::clone(&materialized_schema)),
    );
    let plan = builder.finish_with_host_sink(projection);

    let baseline_context = backend_context_with_grant(8 * 1024 * 1024 * 1024);
    let baseline = execute_query_plan(&baseline_context, &plan).unwrap();
    let baseline_metrics = baseline.metrics();
    let source_read_bytes = baseline_metrics
        .reservation_events
        .iter()
        .filter(|event| {
            event.outcome.as_str() == "granted"
                && event.reservation_kind == ReservationKind::SourceRead
                && event.memory_boundary == "source_chunk_read"
        })
        .map(|event| event.requested_bytes)
        .sum::<u64>();
    let chunk_operator_bytes =
        granted_reservation_requested_bytes(baseline_metrics, ReservationKind::ChunkOperator, "chunk_operator");
    let host_materialization_bytes =
        granted_reservation_requested_bytes(baseline_metrics, ReservationKind::Materialization, "host_materialization");
    let execution_peak = source_read_bytes.saturating_add(chunk_operator_bytes);
    let constrained_budget = execution_peak.max(host_materialization_bytes).saturating_add(4096);
    assert!(
        constrained_budget < host_materialization_bytes.saturating_mul(2),
        "test data must leave a budget window where execution fits but materialization reservation fails"
    );

    drop(baseline_context);

    let constrained_context = backend_context_with_grant(constrained_budget);

    let error = match execute_query_plan(&constrained_context, &plan) {
        Ok(_) => panic!("bounded execution should fail the host materialization reservation"),
        Err(error) => error,
    };
    assert_eq!(error.code(), QueryErrorCode::MemoryReservationFailed);
    assert_eq!(error.fact_value("memory_boundary"), Some("host_materialization"));
    assert_eq!(error.fact_value("reservation_kind"), Some("materialization"));
    assert_eq!(error.fact_value("node_id"), Some("n2"));
    assert_eq!(error.fact_value("stage"), Some("materialization"));
    assert_eq!(error.fact_value("attempt"), Some("1"));
    assert_eq!(error.fact_value("max_attempts"), Some("1"));
    assert_eq!(error.fact_value("resume_boundary"), Some("host_materialization"));
    assert_eq!(error.fact_value("committed_chunks"), Some("1"));
    assert_eq!(error.fact_value("committed_partials"), Some("1"));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_executes_non_aggregate_source_plan() {
    let dir = unique_temp_dir("nexus-query-engine-bounded-source-only");
    let path = dir.join("input.parquet");
    let schema = write_row_grouped_input_parquet(&path);
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&schema),
        vec!["key".into(), "payload".into(), "m00".into(), "m01".into()],
    );

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let plan = builder.finish_with_host_sink(source);
    let context = backend_context_with_grant(1024 * 1024);

    let result = execute_query_plan(&context, &plan).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should return a RecordBatch");
    };
    assert_eq!(batch.num_rows(), 6);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[1, 2, 1, 3, 2, 3]);
    assert_eq!(int64_array(batch, 1).values().as_ref(), &[10, 10, 20, 20, 30, 5]);
    assert_eq!(result.metrics().parquet_files, 1);
    assert!(result.metrics().reservation_granted_bytes > 0);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_preserves_multi_file_source_order() {
    let dir = unique_temp_dir("nexus-query-engine-bounded-multi-file-source-order");
    let first_path = dir.join("part-000.parquet");
    let second_path = dir.join("part-001.parquet");
    let third_path = dir.join("part-002.parquet");
    let schema = write_i32_parquet_with_row_group_rows(&first_path, &[1, 2], 1);
    write_i32_parquet_with_row_group_rows(&second_path, &[10, 11], 1);
    write_i32_parquet_with_row_group_rows(&third_path, &[20], 1);
    let source = LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new(&second_path)
                .with_file_size_bytes(fs::metadata(&second_path).unwrap().len())
                .with_row_group_rows(vec![1, 1]),
            LocalParquetFile::new(&first_path)
                .with_file_size_bytes(fs::metadata(&first_path).unwrap().len())
                .with_row_group_rows(vec![1, 1]),
            LocalParquetFile::new(&third_path)
                .with_file_size_bytes(fs::metadata(&third_path).unwrap().len())
                .with_row_group_rows(vec![1]),
        ],
        Arc::clone(&schema),
        vec!["id".to_string()],
    );

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let plan = builder.finish_with_host_sink(source);
    let context = backend_context_with_grant(1024 * 1024);

    let result = execute_query_plan(&context, &plan).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should return a RecordBatch");
    };
    assert_eq!(batch.num_rows(), 5);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[10, 11, 1, 2, 20]);
    assert_eq!(result.metrics().parquet_files, 3);
    assert_eq!(result.metrics().source_chunks_committed, 5);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_executes_union_plan_without_legacy_aggregate_path() {
    let context = backend_context_with_grant(1024 * 1024);
    let runtime = Arc::new(context.runtime().unwrap());
    let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, false)]));
    let left_batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from(vec![1, 2]))]).unwrap();
    let right_batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from(vec![3, 4]))]).unwrap();

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("left", Arc::clone(&schema))));
    let right = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("right", Arc::clone(&schema))));
    let union = builder.add_union(vec![left, right], Arc::clone(&schema));
    let plan = builder.finish_with_host_sink(union);
    let mut gpu_inputs = GpuResidentInputMap::new();
    gpu_inputs.insert(
        "left".to_string(),
        GpuResultHandle::new(
            &context,
            GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &left_batch).unwrap()),
            Arc::clone(&schema),
        )
        .unwrap(),
    );
    gpu_inputs.insert(
        "right".to_string(),
        GpuResultHandle::new(
            &context,
            GpuDataFrame::new(DataFrame::from_record_batch(&runtime, &right_batch).unwrap()),
            Arc::clone(&schema),
        )
        .unwrap(),
    );

    let result = execute_query_plan_with_gpu_inputs(&context, &plan, &gpu_inputs).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should return a RecordBatch");
    };
    assert_eq!(batch.num_rows(), 4);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[1, 2, 3, 4]);
    assert_eq!(result.metrics().reservation_denied_count, 0);
}
