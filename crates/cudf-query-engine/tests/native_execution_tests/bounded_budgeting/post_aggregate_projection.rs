use super::*;

#[test]
fn bounded_native_query_executes_post_aggregate_projection_before_final_sort() {
    let dir = unique_temp_dir("cudf-query-engine-bounded-post-aggregate-projection");
    let path = dir.join("input.parquet");
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let input_schema = write_row_grouped_input_parquet(&path);
    let aggregated_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("sum_m00", DataType::Int64, true),
    ]));
    let post_projection_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("total", DataType::Int64, true),
    ]));

    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["key".to_string(), "payload".to_string(), "m00".to_string()],
    );
    let filter = Expr::binary(Expr::column("payload"), BinaryOp::GtEq, Expr::literal(LiteralValue::Int64(10)));
    let aggregate = AggregateSpec::new(
        vec![Expr::column("key")],
        vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("m00"), "sum_m00")],
    );
    let projection = vec![Expr::column("key"), Expr::column("sum_m00").alias("total")];
    let sort = SortSpec::new(vec![
        SortExpr::new(Expr::column("total"), true, false),
        SortExpr::new(Expr::column("key"), false, false),
    ]);

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let filter = builder.add_filter(source, filter, Arc::clone(&input_schema));
    let aggregate = builder.add_aggregate(filter, aggregate, aggregated_schema);
    let projection =
        builder.add_projection(aggregate, projection_outputs(projection, Arc::clone(&post_projection_schema)));
    let sort = builder.add_sort(projection, sort, Arc::clone(&post_projection_schema));
    let limit = builder.add_limit(sort, LimitSpec::fetch(3), post_projection_schema);
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
    assert!(metrics.reservation_events.iter().any(|event| {
        event.outcome.as_str() == "granted"
            && event.reservation_kind == ReservationKind::FinalMerge
            && event.memory_boundary == "post_aggregate_projection"
    }));
    assert!(metrics.reservation_events.iter().any(|event| {
        event.outcome.as_str() == "granted"
            && event.reservation_kind == ReservationKind::FinalMerge
            && event.memory_boundary == "post_aggregate_projection_output"
    }));
    assert_budgeted_success_event_with_contract(
        metrics,
        "cudf_query_engine.projection.execute",
        3,
        "post_aggregate_projection",
        Some("libcudf_dataframe_expression_temporary_workspace_and_transient_relocation_preflight_unavailable"),
    );

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 3);
    assert_eq!(batch.schema().field(1).name(), "total");
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[2, 1, 3]);
    assert_eq!(int64_array(batch, 1).values().as_ref(), &[700, 400, 400]);

    fs::remove_dir_all(dir).unwrap();
}
