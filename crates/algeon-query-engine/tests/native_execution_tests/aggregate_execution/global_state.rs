use std::sync::Arc;

use super::*;

#[test]
fn bounded_native_query_executes_global_sum_min_max_count_and_final_merge() {
    let dir = unique_temp_dir("query-engine-bounded-native-global-aggregate-parity");
    let path = dir.join("input.parquet");
    let context = backend_context_with_memory_control_policy(MemoryControlPolicy {
        device_budget_bytes: 8 * 1024 * 1024 * 1024,
        max_source_chunk_bytes: 1024 * 1024,
        max_row_groups_per_chunk: 1,
        max_retry_attempts: 1,
        min_retry_chunk_row_groups: 1,
    });
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

    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.source_chunks_planned, 3);
    assert_eq!(metrics.source_chunks_committed, 3);
    assert_eq!(metrics.partial_aggregate_count, 1);
    assert_eq!(metrics.final_merge_attempts, 0);
    assert_eq!(metrics.boundary.host_to_device_uploads, 0);
    assert_eq!(metrics.boundary.device_to_host_materializations, 1);
    assert_budgeted_success_event_with_boundary(
        metrics,
        "query_engine.global_aggregate.count_output",
        1,
        "chunk_operator",
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
    let dir = unique_temp_dir("query-engine-bounded-native-single-chunk-global-state");
    let path = dir.join("input.parquet");
    let context = backend_context_with_memory_control_policy(MemoryControlPolicy {
        device_budget_bytes: 8 * 1024 * 1024 * 1024,
        max_source_chunk_bytes: 1024 * 1024,
        max_row_groups_per_chunk: 8,
        max_retry_attempts: 1,
        min_retry_chunk_row_groups: 1,
    });
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

    let host_plan = build_plan(true);
    let host_admitted = context.admit_plan(&host_plan).unwrap();
    {
        let host_result =
            execute_query_plan_with_gpu_inputs(&host_admitted, &host_plan, &GpuResidentInputMap::new()).unwrap();
        assert_eq!(host_result.metrics().source_chunks_planned, 1);
        assert_eq!(host_result.metrics().partial_aggregate_count, 1);
        assert_eq!(host_result.metrics().final_merge_attempts, 0);
        let QueryOutput::HostArrow(batch) = host_result.output() else {
            panic!("host sink should materialize one RecordBatch");
        };
        assert_eq!(batch.num_columns(), expected_names.len());
        assert_eq!(
            batch.schema().fields().iter().map(|field| field.name().as_str()).collect::<Vec<_>>(),
            expected_names
        );
        assert!(
            batch
                .schema()
                .fields()
                .iter()
                .all(|field| !field.name().starts_with("__algeon_native_global_valid_count_"))
        );
        assert_eq!(batch.num_rows(), 1);
        assert_eq!(int64_array(batch, 0).values().as_ref(), &[1500]);
        assert_eq!(int64_array(batch, 1).values().as_ref(), &[100]);
        assert_eq!(int64_array(batch, 2).values().as_ref(), &[5]);
        assert_eq!(int64_array(batch, 3).values().as_ref(), &[5]);
    }
    host_admitted.finish();
    drop(host_admitted);

    let gpu_plan = build_plan(false);
    let gpu_result = execute_query_plan(&context, &gpu_plan).unwrap();
    assert_eq!(gpu_result.metrics().source_chunks_planned, 1);
    assert_eq!(gpu_result.metrics().partial_aggregate_count, 1);
    assert_eq!(gpu_result.metrics().final_merge_attempts, 0);
    let QueryOutput::GpuResident(handle) = gpu_result.output() else {
        panic!("GPU sink should retain the finalized global aggregate");
    };
    let mut materialization_metrics = QueryMetrics::default();
    let batch = materialize_gpu_result_to_host(handle.clone(), &mut materialization_metrics).unwrap();
    assert_eq!(batch.num_columns(), expected_names.len());
    assert_eq!(batch.num_rows(), 1);
    assert_eq!(int64_array(&batch, 0).values().as_ref(), &[1500]);
    assert_eq!(int64_array(&batch, 1).values().as_ref(), &[100]);
    assert_eq!(int64_array(&batch, 2).values().as_ref(), &[5]);
    assert_eq!(int64_array(&batch, 3).values().as_ref(), &[5]);
    drop(gpu_result);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn native_query_rejects_global_valid_count_reserved_state_alias_collision_before_source_read() {
    let context = backend_context();
    let input_schema = Arc::new(Schema::new(vec![Field::new("m00", DataType::Int64, false)]));
    let reserved_alias = "__algeon_native_global_valid_count_sum_m00";
    let aggregated_schema = Arc::new(Schema::new(vec![
        Field::new("sum_m00", DataType::Int64, true),
        Field::new(reserved_alias, DataType::Int64, false),
    ]));
    let aggregate = AggregateSpec::new(
        Vec::new(),
        vec![
            AggregateExpr::new(AggregateFunction::Sum, Expr::column("m00"), "sum_m00"),
            AggregateExpr::new(AggregateFunction::Count, Expr::column("m00"), reserved_alias),
        ],
    );

    let mut builder = QueryPlanBuilder::new();
    let source =
        builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("missing_input", Arc::clone(&input_schema))));
    let aggregate = builder.add_aggregate(source, aggregate, aggregated_schema);
    let plan = builder.finish_with_host_sink(aggregate);

    match execute_query_plan(&context, &plan) {
        Ok(_) => panic!("global valid-count reserved state alias collision should be rejected before source read"),
        Err(error) => {
            assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
            assert_eq!(error.operation(), "query_engine.global_aggregate.validate_valid_count_state_alias");
            assert_eq!(error.fact_value("alias"), Some(reserved_alias));
            assert_eq!(error.fact_value("reserved_aliases"), Some(reserved_alias));
        },
    }
}
