use std::sync::Arc;

use super::*;

#[test]
fn bounded_native_query_executes_filtered_decimal_global_sum() {
    let dir = unique_temp_dir("query-engine-bounded-filtered-decimal-sum");
    let path = dir.join("lineitem.parquet");
    let context = backend_context_with_memory_control_policy(MemoryControlPolicy {
        device_budget_bytes: 8 * 1024 * 1024 * 1024,
        max_source_chunk_bytes: 1024 * 1024,
        max_row_groups_per_chunk: 1,
        max_retry_attempts: 1,
        min_retry_chunk_row_groups: 1,
    });
    let input_schema = write_discounted_lineitem_parquet(&path);
    let projected_schema =
        Arc::new(Schema::new(vec![Field::new("__algeon_native_aggregate_arg_0", DataType::Decimal128(38, 4), true)]));
    let aggregated_schema = Arc::new(Schema::new(vec![Field::new("revenue", DataType::Decimal128(38, 4), true)]));

    let source = LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&path).with_file_size_bytes(fs::metadata(&path).unwrap().len())],
        Arc::clone(&input_schema),
        vec![
            "l_discount".to_string(),
            "l_extendedprice".to_string(),
            "l_quantity".to_string(),
            "l_shipdate".to_string(),
        ],
    );
    let filter = Expr::binary(
        Expr::binary(
            Expr::binary(
                Expr::binary(
                    Expr::column("l_shipdate"),
                    BinaryOp::GtEq,
                    Expr::literal(LiteralValue::Date32(date32(1994, 1, 1))),
                ),
                BinaryOp::And,
                Expr::binary(
                    Expr::column("l_shipdate"),
                    BinaryOp::Lt,
                    Expr::literal(LiteralValue::Date32(date32(1995, 1, 1))),
                ),
            ),
            BinaryOp::And,
            Expr::binary(
                Expr::binary(
                    Expr::column("l_discount"),
                    BinaryOp::GtEq,
                    Expr::literal(LiteralValue::Decimal128 { raw: 5, precision: 15, scale: 2 }),
                ),
                BinaryOp::And,
                Expr::binary(
                    Expr::column("l_discount"),
                    BinaryOp::LtEq,
                    Expr::literal(LiteralValue::Decimal128 { raw: 7, precision: 15, scale: 2 }),
                ),
            ),
        ),
        BinaryOp::And,
        Expr::binary(
            Expr::column("l_quantity"),
            BinaryOp::Lt,
            Expr::literal(LiteralValue::Decimal128 { raw: 2_400, precision: 15, scale: 2 }),
        ),
    );
    let projection = vec![
        Expr::binary(Expr::column("l_extendedprice"), BinaryOp::Multiply, Expr::column("l_discount"))
            .alias("__algeon_native_aggregate_arg_0"),
    ];
    let aggregate = AggregateSpec::new(
        Vec::new(),
        vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("__algeon_native_aggregate_arg_0"), "revenue")],
    );

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let filter = builder.add_filter(source, filter).unwrap();
    let projection = builder.add_projection(filter, projection_outputs(projection, projected_schema)).unwrap();
    let aggregate = builder.add_aggregate(projection, aggregate, aggregated_schema).unwrap();
    let plan = builder.finish_with_host_sink(aggregate).unwrap();

    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.source_chunks_planned, 4);
    assert_eq!(metrics.source_chunks_committed, 4);
    assert_eq!(metrics.partial_aggregate_count, 1);
    assert_eq!(metrics.final_merge_attempts, 0);
    assert_eq!(metrics.boundary.host_to_device_uploads, 0);
    assert_eq!(metrics.boundary.device_to_host_materializations, 1);
    assert!(
        budgeted_events(metrics, "query_engine.global_aggregate.cast_input").is_empty(),
        "decimal SUM should not materialize a redundant input cast when the input already has the target dtype"
    );

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 1);
    let revenue = decimal128_array(batch, 0);
    assert_eq!(revenue.value(0), 260_000);
    assert_eq!(revenue.scale(), 4);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_zero_column_count_uses_row_group_metadata_without_read_columns() {
    let context = backend_context_with_memory_control_policy(MemoryControlPolicy {
        device_budget_bytes: 8 * 1024 * 1024 * 1024,
        max_source_chunk_bytes: 250,
        max_row_groups_per_chunk: 8,
        max_retry_attempts: 1,
        min_retry_chunk_row_groups: 1,
    });
    let aggregated_schema = Arc::new(Schema::new(vec![Field::new("count_rows", DataType::Int64, false)]));
    let source = LocalParquetSource::zero_column_count(vec![
        LocalParquetFile::new("/tmp/count-only.parquet")
            .with_stable_id("count-only")
            .with_file_size_bytes(600)
            .with_row_groups(vec![0, 2])
            .with_row_group_bytes(vec![100, 200, 300])
            .with_row_group_rows(vec![5, 7, 11]),
    ]);
    let aggregate = AggregateSpec::new(
        Vec::new(),
        vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column(ZERO_COLUMN_COUNT_COLUMN), "count_rows")],
    );
    let sort = SortSpec::new(vec![SortExpr::new(Expr::column("count_rows"), true, false)]);

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let aggregate = builder.add_aggregate(source, aggregate, Arc::clone(&aggregated_schema)).unwrap();
    let sort = builder.add_sort(aggregate, sort).unwrap();
    let limit = builder.add_limit(sort, LimitSpec::fetch(1)).unwrap();
    let plan = builder.finish_with_host_sink(limit).unwrap();

    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.source_chunks_planned, 2);
    assert_eq!(metrics.source_chunks_committed, 2);
    assert_eq!(metrics.row_groups_selected, 2);
    assert_eq!(metrics.read_columns, 0);
    assert_eq!(metrics.input_rows, 16);
    assert_eq!(metrics.boundary.host_to_device_uploads, 0);
    assert_eq!(metrics.boundary.device_to_host_materializations, 1);
    assert_budgeted_success_event_with_rows(metrics, "query_engine.local_parquet_chunk.zero_column_count", 1);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 1);
    assert_eq!(int64_array(batch, 0).values().as_ref(), &[16]);
}
