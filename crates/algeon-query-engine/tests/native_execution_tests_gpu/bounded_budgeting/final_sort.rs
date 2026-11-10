use std::sync::Arc;

use super::*;

#[test]
fn bounded_native_query_uses_hierarchical_topk_without_full_sort_first() {
    let dir = unique_temp_dir("query-engine-bounded-final-sort-proactive-topk");
    let path = dir.join("input.parquet");
    let input_schema = write_single_row_group_final_sort_input_parquet(&path, 4096);
    let aggregated_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("sum_m00", DataType::Int64, true),
    ]));
    let final_sort_schema = wide_final_sort_schema(16);

    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["key".to_string(), "m00".to_string()],
    );
    let aggregate = AggregateSpec::new(
        vec![Expr::column("key")],
        vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("m00"), "sum_m00")],
    );
    let sort = SortSpec::new(vec![SortExpr::new(Expr::column("key"), true, false)]);
    let limit = LimitSpec::new(3, 7);

    let mut builder = QueryPlanBuilder::new();
    let source_id = builder.add_source(SourceNode::LocalParquet(source.clone()));
    let one_chunk_plan =
        plan_local_parquet_source_chunks(source_id, &source, SourceChunkPlanningPolicy::new(None, 1)).unwrap();
    assert_eq!(one_chunk_plan.chunk_count(), 1);
    let aggregate = builder.add_aggregate(source_id, aggregate, Arc::clone(&aggregated_schema)).unwrap();
    let projection = builder
        .add_projection(aggregate, projection_outputs(wide_final_sort_projection(16), Arc::clone(&final_sort_schema)))
        .unwrap();
    let sort = builder.add_sort(projection, sort).unwrap();
    let limit = builder.add_limit(sort, limit).unwrap();
    let plan = builder.finish_with_host_sink(limit).unwrap();

    let context = backend_context_with_memory_control_policy(MemoryControlPolicy {
        device_budget_bytes: 1_200_000,
        max_source_chunk_bytes: 1024 * 1024,
        max_row_groups_per_chunk: 1,
        max_retry_attempts: 1,
        min_retry_chunk_row_groups: 1,
    });
    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.source_chunks_planned, 1);
    assert_eq!(metrics.source_chunks_committed, 1);
    assert_eq!(metrics.retry_count, 0);
    assert!(
        metrics.budgeted_execution_events.iter().all(|event| event.operation != "query_engine.final_sort.prepare"
            && event.operation != "query_engine.final_sort.materialize"),
        "selective ORDER BY LIMIT should not run full final sort before top-k"
    );
    assert_final_sort_prepare_budget_event(
        metrics,
        "query_engine.final_sort.topk_prepare",
        "final_sort_topk_stream_chunk",
        None,
    );
    assert_final_sort_materialize_budget_event(
        metrics,
        "query_engine.final_sort.topk_materialize",
        "final_sort_topk_stream_chunk",
        None,
    );
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 7);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[4092, 4091, 4090, 4089, 4088, 4087, 4086]);
    assert_eq!(int64_array(batch, 1).values().as_ref(), &[1, 1, 1, 1, 1, 1, 1]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_full_sort_without_limit_fits_a_cap_below_input_plus_output() {
    let rows = 1 << 20;
    let computed_columns = 7;
    let dir = unique_temp_dir("query-engine-bounded-full-sort-releases-input");
    let path = dir.join("input.parquet");
    let input_schema = write_single_row_group_final_sort_input_parquet(&path, rows);
    let mut fields = vec![Field::new("key", DataType::Int32, false)];
    let mut projection = vec![Expr::column("key")];
    for index in 0..computed_columns {
        fields.push(Field::new(format!("x{index}"), DataType::Int64, true));
        projection.push(
            Expr::binary(Expr::column("m00"), BinaryOp::Add, Expr::literal(LiteralValue::Int64(index)))
                .alias(format!("x{index}")),
        );
    }
    let sorted_schema = Arc::new(Schema::new(fields));
    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec!["key".to_string(), "m00".to_string()],
    );

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let projection = builder.add_projection(source, projection_outputs(projection, sorted_schema)).unwrap();
    let sort =
        builder.add_sort(projection, SortSpec::new(vec![SortExpr::new(Expr::column("key"), true, false)])).unwrap();
    let plan = builder.finish_with_host_sink(sort).unwrap();

    // The sort input is 60 bytes per row and its order 4. Holding the input while the
    // whole output is gathered needs about 124 bytes per row; releasing each input column
    // after its sorted copy needs about 72.
    let context = backend_context_with_memory_control_policy(MemoryControlPolicy {
        device_budget_bytes: 100 * rows as u64,
        max_source_chunk_bytes: 1024 * 1024 * 1024,
        max_row_groups_per_chunk: 1,
        max_retry_attempts: 1,
        min_retry_chunk_row_groups: 1,
    });
    let result = execute_query_plan(&context, &plan).unwrap();
    assert_eq!(result.metrics().retry_count, 0);
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), rows as usize);
    let keys = int32_array(batch, 0);
    assert_eq!(keys.value(0), rows - 1);
    assert_eq!(keys.value(rows as usize - 1), 0);
    assert!(int64_array(batch, computed_columns as usize).values().iter().all(|&value| value == computed_columns));

    fs::remove_dir_all(dir).unwrap();
}
