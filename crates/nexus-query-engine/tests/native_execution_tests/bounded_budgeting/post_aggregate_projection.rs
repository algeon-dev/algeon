use std::sync::Arc;

use super::*;

#[test]
fn bounded_native_query_executes_post_aggregate_projection_before_final_sort() {
    let dir = unique_temp_dir("nexus-query-engine-bounded-post-aggregate-projection");
    let path = dir.join("input.parquet");
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

    let context = backend_context_with_grant(8 * 1024 * 1024 * 1024);
    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert!(metrics.reservation_events.iter().any(|event| {
        event.outcome.as_str() == "granted"
            && event.reservation_kind == ReservationKind::FinalMerge
            && event.memory_boundary == "post_aggregate_projection"
    }));
    assert!(metrics.reservation_events.iter().any(|event| {
        event.reservation_kind == ReservationKind::FinalMerge
            && event.memory_boundary == "post_aggregate_projection_output"
            && event.action == MemoryReservationEventAction::Transfer
    }));
    assert!(
        metrics
            .budgeted_execution_events
            .iter()
            .all(|event| event.operation != "nexus_query_engine.projection.execute"),
        "schema-only post-aggregate projection should use the column-select fast path instead of expression workspace: {:?}",
        metrics
            .budgeted_execution_events
            .iter()
            .filter(|event| event.operation == "nexus_query_engine.projection.execute")
            .collect::<Vec<_>>()
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

#[test]
fn bounded_native_query_executes_filter_between_aggregate_and_sort() {
    let dir = unique_temp_dir("nexus-query-engine-bounded-post-aggregate-filter");
    let path = dir.join("input.parquet");
    let runtime = backend_runtime();
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
        vec!["key".to_string(), "m00".to_string()],
    );
    let aggregate = AggregateSpec::new(
        vec![Expr::column("key")],
        vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("m00"), "sum_m00")],
    );
    let post_filter = Expr::binary(Expr::column("sum_m00"), BinaryOp::Gt, Expr::literal(LiteralValue::Int64(650)));
    let projection = vec![Expr::column("key"), Expr::column("sum_m00").alias("total")];
    let sort = SortSpec::new(vec![SortExpr::new(Expr::column("total"), true, false)]);

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let aggregate = builder.add_aggregate(source, aggregate, Arc::clone(&aggregated_schema));
    let filter = builder.add_filter(aggregate, post_filter, Arc::clone(&aggregated_schema));
    let projection =
        builder.add_projection(filter, projection_outputs(projection, Arc::clone(&post_projection_schema)));
    let sort = builder.add_sort(projection, sort, Arc::clone(&post_projection_schema));
    let limit = builder.add_limit(sort, LimitSpec::fetch(25), post_projection_schema);
    let plan = builder.finish_with_host_sink(limit);

    let context = bounded_test_context(runtime);
    let result = execute_query_plan(&context, &plan).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert!(batch.num_rows() >= 1);
    assert!(int64_array(batch, 1).values().iter().all(|value| *value > 650));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_executes_projection_between_sort_and_limit() {
    let dir = unique_temp_dir("nexus-query-engine-bounded-post-sort-projection");
    let path = dir.join("input.parquet");
    let runtime = backend_runtime();
    let input_schema = write_row_grouped_input_parquet(&path);
    let aggregated_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("sum_m00", DataType::Int64, true),
    ]));
    let sort_input_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("total", DataType::Int64, true),
        Field::new("sort_key", DataType::Int64, true),
    ]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("total", DataType::Int64, true),
    ]));

    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["key".to_string(), "m00".to_string()],
    );
    let aggregate = AggregateSpec::new(
        vec![Expr::column("key")],
        vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("m00"), "sum_m00")],
    );
    let sort_input_projection =
        vec![Expr::column("key"), Expr::column("sum_m00").alias("total"), Expr::column("sum_m00").alias("sort_key")];
    let output_projection = vec![Expr::column("key"), Expr::column("total")];
    let sort = SortSpec::new(vec![
        SortExpr::new(Expr::column("sort_key"), true, false),
        SortExpr::new(Expr::column("key"), false, false),
    ]);

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let aggregate = builder.add_aggregate(source, aggregate, aggregated_schema);
    let sort_input =
        builder.add_projection(aggregate, projection_outputs(sort_input_projection, Arc::clone(&sort_input_schema)));
    let sort = builder.add_sort(sort_input, sort, Arc::clone(&sort_input_schema));
    let output = builder.add_projection(sort, projection_outputs(output_projection, Arc::clone(&output_schema)));
    let limit = builder.add_limit(output, LimitSpec::fetch(2), output_schema);
    let plan = builder.finish_with_host_sink(limit);

    let context = bounded_test_context(runtime);
    let result = execute_query_plan(&context, &plan).unwrap();
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 2);
    assert_eq!(batch.num_columns(), 2);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[3, 2]);

    fs::remove_dir_all(dir).unwrap();
}

fn bounded_test_context(_runtime: Arc<cudf::Runtime>) -> BackendQueryContext {
    backend_context_with_grant(8 * 1024 * 1024 * 1024_u64)
}
