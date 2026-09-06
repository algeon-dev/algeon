use std::{fs::File, path::Path, sync::Arc};

use nexus_query_engine::mode::StatisticalAggregateExecutionMode;

use super::*;

type RollupRankRow = (Option<String>, Option<String>, Option<String>, u32, i64, u64);

#[test]
fn bounded_native_query_executes_root_hash_join_under_aggregate_with_ordered_reservations() {
    let runtime = backend_runtime();
    let dir = unique_temp_dir("nexus-query-engine-bounded-root-join-aggregate");
    let left_path = dir.join("lineitem.parquet");
    let right_path = dir.join("orders.parquet");
    let left_schema = write_join_left_parquet(&runtime, &left_path);
    let right_schema = write_join_right_parquet(&runtime, &right_path);
    let join_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
    ]));
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("line_count", DataType::Int64, false),
    ]));
    let sort = SortSpec::new(vec![SortExpr::new(Expr::column("order_id"), false, false)]);

    let mut builder = QueryPlanBuilder::new();
    let left =
        builder.add_source(SourceNode::LocalParquet(join_left_source(&left_path, &left_schema, &["order_id", "qty"])));
    let right =
        builder.add_source(SourceNode::LocalParquet(join_right_source(&right_path, &right_schema, &["order_id"])));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(JoinType::Inner, vec![join_order_id_key()])
            .with_output_projection(vec![
                JoinOutputColumn::left(
                    nexus_query_engine::expr::ColumnRef::input(0, "order_id", "order_id"),
                    "order_id",
                ),
                JoinOutputColumn::left(nexus_query_engine::expr::ColumnRef::input(0, "qty", "qty"), "qty"),
            ])
            .with_build_side(JoinSide::Right),
        join_schema,
    );
    let aggregate = builder.add_aggregate(
        join,
        AggregateSpec::new(
            vec![Expr::column("order_id")],
            vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("qty"), "line_count")],
        ),
        Arc::clone(&aggregate_schema),
    );
    let sort = builder.add_sort(aggregate, sort, Arc::clone(&aggregate_schema));
    let limit = builder.add_limit(sort, LimitSpec::fetch(10), aggregate_schema);
    let plan = builder.finish_with_host_sink(limit);
    let context = backend_context_with_grant(1024 * 1024);

    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.memory_control_policy.as_str(), "bounded");
    assert_eq!(metrics.hash_join_build_count, 1);
    assert_eq!(metrics.compiled_pipeline_attempts, 1);
    assert!(metrics.compiled_pipeline_tasks_executed > 0);
    assert_eq!(metrics.compiled_pipeline_host_arrow_sinks, 1);
    assert_eq!(metrics.partial_aggregate_count, 1);
    assert_eq!(metrics.final_merge_attempts, 0);
    assert_eq!(metrics.boundary.device_to_host_materializations, 1);
    let join_metrics =
        metrics.operators.iter().find_map(|operator| operator.join.as_ref()).expect("join metrics should be recorded");
    assert_eq!(join_metrics.join_type, "inner");
    assert_eq!(join_metrics.output_rows, 3);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 2);
    assert_eq!(batch.num_columns(), 2);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[2, 3]);
    assert_eq!(int64_array(batch, 1).values().as_ref(), &[2, 1]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_fuses_compiled_sort_limit_with_topk() {
    let runtime = backend_runtime();
    let dir = unique_temp_dir("nexus-query-engine-bounded-compiled-sort-limit-topk");
    let left_path = dir.join("lineitem.parquet");
    let right_path = dir.join("orders.parquet");
    let left_schema = write_join_left_parquet(&runtime, &left_path);
    let right_schema = write_join_right_parquet(&runtime, &right_path);
    let join_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
    ]));
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("line_count", DataType::Int64, false),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let left =
        builder.add_source(SourceNode::LocalParquet(join_left_source(&left_path, &left_schema, &["order_id", "qty"])));
    let right =
        builder.add_source(SourceNode::LocalParquet(join_right_source(&right_path, &right_schema, &["order_id"])));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(JoinType::Inner, vec![join_order_id_key()])
            .with_output_projection(vec![
                JoinOutputColumn::left(
                    nexus_query_engine::expr::ColumnRef::input(0, "order_id", "order_id"),
                    "order_id",
                ),
                JoinOutputColumn::left(nexus_query_engine::expr::ColumnRef::input(0, "qty", "qty"), "qty"),
            ])
            .with_build_side(JoinSide::Right),
        join_schema,
    );
    let aggregate = builder.add_aggregate(
        join,
        AggregateSpec::new(
            vec![Expr::column("order_id")],
            vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("qty"), "line_count")],
        ),
        Arc::clone(&aggregate_schema),
    );
    let sort = builder.add_sort(
        aggregate,
        SortSpec::new(vec![
            SortExpr::new(Expr::column("line_count"), true, false),
            SortExpr::new(Expr::column("order_id"), false, false),
        ]),
        Arc::clone(&aggregate_schema),
    );
    let limit = builder.add_limit(sort, LimitSpec::fetch(1), aggregate_schema);
    let plan = builder.finish_with_host_sink(limit);
    let context = backend_context_with_grant(1024 * 1024);

    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert!(
        metrics
            .budgeted_execution_events
            .iter()
            .any(|event| event.operation == "nexus_query_engine.final_sort.topk_prepare"
                && event.boundary == "final_sort_topk_stream_chunk"),
        "selective compiled Sort+Limit should top-k the lazily finalized aggregate output chunk by chunk"
    );
    assert!(
        metrics
            .budgeted_execution_events
            .iter()
            .all(|event| event.operation != "nexus_query_engine.final_sort.prepare"),
        "selective compiled Sort+Limit should not full-sort before top-k"
    );

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 1);
    assert_eq!(int32_array(batch, 0).values().as_ref(), &[2]);
    assert_eq!(int64_array(batch, 1).values().as_ref(), &[2]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_executes_serial_aggregate_join_spine() {
    let runtime = backend_runtime();
    let dir = unique_temp_dir("nexus-query-engine-bounded-serial-aggregate-spine");
    let customer_path = dir.join("customer.parquet");
    let orders_path = dir.join("orders.parquet");
    let customer_schema = write_customer_order_count_customer_parquet(&runtime, &customer_path);
    let orders_schema = write_customer_order_count_orders_parquet(&runtime, &orders_path);
    let join_schema = Arc::new(Schema::new(vec![
        Field::new("c_custkey", DataType::Int32, false),
        Field::new("o_orderkey", DataType::Int32, true),
    ]));
    let customer_count_schema = Arc::new(Schema::new(vec![
        Field::new("c_custkey", DataType::Int32, false),
        Field::new("c_count", DataType::Int64, false),
    ]));
    let c_count_schema = Arc::new(Schema::new(vec![Field::new("c_count", DataType::Int64, false)]));
    let final_schema = Arc::new(Schema::new(vec![
        Field::new("c_count", DataType::Int64, false),
        Field::new("custdist", DataType::Int64, false),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let customers = builder.add_source(SourceNode::LocalParquet(join_left_source(
        &customer_path,
        &customer_schema,
        &["c_custkey"],
    )));
    let orders = builder.add_source(SourceNode::LocalParquet(join_right_source(
        &orders_path,
        &orders_schema,
        &["o_orderkey", "o_custkey"],
    )));
    let join = builder.add_join(
        customers,
        orders,
        JoinSpec::new(
            JoinType::Left,
            vec![JoinKey::new(
                nexus_query_engine::expr::ColumnRef::input(0, "c_custkey", "c_custkey"),
                nexus_query_engine::expr::ColumnRef::input(1, "o_custkey", "o_custkey"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(
                nexus_query_engine::expr::ColumnRef::input(0, "c_custkey", "c_custkey"),
                "c_custkey",
            ),
            JoinOutputColumn::right(
                nexus_query_engine::expr::ColumnRef::input(1, "o_orderkey", "o_orderkey"),
                "o_orderkey",
            ),
        ])
        .with_build_side(JoinSide::Right),
        Arc::clone(&join_schema),
    );
    let projection = builder.add_projection(
        join,
        projection_outputs(vec![Expr::column("c_custkey"), Expr::column("o_orderkey")], Arc::clone(&join_schema)),
    );
    let customer_counts = builder.add_aggregate(
        projection,
        AggregateSpec::new(
            vec![Expr::column("c_custkey")],
            vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("o_orderkey"), "c_count")],
        ),
        customer_count_schema,
    );
    let count_projection =
        builder.add_projection(customer_counts, projection_outputs(vec![Expr::column("c_count")], c_count_schema));
    let distribution = builder.add_aggregate(
        count_projection,
        AggregateSpec::new(
            vec![Expr::column("c_count")],
            vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("c_count"), "custdist")],
        ),
        Arc::clone(&final_schema),
    );
    let sort = builder.add_sort(
        distribution,
        SortSpec::new(vec![
            SortExpr::new(Expr::column("custdist"), true, false),
            SortExpr::new(Expr::column("c_count"), true, false),
        ]),
        final_schema,
    );
    let plan = builder.finish_with_host_sink(sort);
    assert!(!plan.capability_analysis().admits_root_join());

    let context = bounded_join_execution_context(&runtime);
    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.memory_control_policy.as_str(), "bounded");
    assert_eq!(metrics.hash_join_build_count, 1);
    assert_eq!(metrics.compiled_pipeline_attempts, 1);
    assert!(metrics.compiled_pipeline_tasks_executed > 0);
    assert_eq!(metrics.compiled_pipeline_host_arrow_sinks, 1);
    assert_eq!(metrics.boundary.device_to_host_materializations, 1);
    let join_metrics =
        metrics.operators.iter().find_map(|operator| operator.join.as_ref()).expect("join metrics should be recorded");
    assert_eq!(join_metrics.join_type, "left");
    assert_eq!(join_metrics.output_rows, 5);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_rows(), 3);
    assert_eq!(batch.num_columns(), 2);
    assert_eq!(int64_array(batch, 0).values().as_ref(), &[0, 2, 1]);
    assert_eq!(int64_array(batch, 1).values().as_ref(), &[2, 1, 1]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_native_query_executes_rollup_rank_topk_after_aggregate_join_pushdown_envelope() {
    let runtime = backend_runtime();
    let dir = unique_temp_dir("nexus-query-engine-bounded-rollup-rank-topk");
    let fact_path = dir.join("fact.parquet");
    let dim_path = dir.join("dim.parquet");
    let fact_schema = write_rollup_rank_fact_parquet(&fact_path);
    let dim_schema = write_rollup_rank_dim_parquet(&dim_path);
    let detail_aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("category", DataType::Utf8, true),
        Field::new("class", DataType::Utf8, true),
        Field::new("brand", DataType::Utf8, true),
        Field::new("detail_sales", DataType::Int64, true),
    ]));
    let join_schema = Arc::new(Schema::new(vec![
        Field::new("category", DataType::Utf8, true),
        Field::new("class", DataType::Utf8, true),
        Field::new("brand", DataType::Utf8, true),
        Field::new("detail_sales", DataType::Int64, true),
    ]));
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("category", DataType::Utf8, true),
        Field::new("class", DataType::Utf8, true),
        Field::new("brand", DataType::Utf8, true),
        Field::new("__grouping_id", DataType::UInt32, false),
        Field::new("sumsales", DataType::Int64, true),
    ]));
    let ranked_schema = Arc::new(Schema::new(vec![
        Field::new("category", DataType::Utf8, true),
        Field::new("class", DataType::Utf8, true),
        Field::new("brand", DataType::Utf8, true),
        Field::new("__grouping_id", DataType::UInt32, false),
        Field::new("sumsales", DataType::Int64, true),
        Field::new("rk", DataType::UInt64, false),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let fact = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_path(
        &fact_path,
        fact_schema,
        vec!["id".into(), "category".into(), "class".into(), "brand".into(), "amount".into()],
    )));
    let detail_aggregate = builder.add_aggregate(
        fact,
        AggregateSpec::new(
            vec![Expr::column("id"), Expr::column("category"), Expr::column("class"), Expr::column("brand")],
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("amount"), "detail_sales")],
        ),
        Arc::clone(&detail_aggregate_schema),
    );
    let dim = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_path(
        &dim_path,
        dim_schema,
        vec!["id".into()],
    )));
    let join = builder.add_join(
        detail_aggregate,
        dim,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                nexus_query_engine::expr::ColumnRef::input(0, "id", "id"),
                nexus_query_engine::expr::ColumnRef::input(1, "id", "id"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(nexus_query_engine::expr::ColumnRef::input(0, "category", "category"), "category"),
            JoinOutputColumn::left(nexus_query_engine::expr::ColumnRef::input(0, "class", "class"), "class"),
            JoinOutputColumn::left(nexus_query_engine::expr::ColumnRef::input(0, "brand", "brand"), "brand"),
            JoinOutputColumn::left(
                nexus_query_engine::expr::ColumnRef::input(0, "detail_sales", "detail_sales"),
                "detail_sales",
            ),
        ])
        .with_build_side(JoinSide::Right),
        Arc::clone(&join_schema),
    );
    let aggregate = builder.add_aggregate(
        join,
        AggregateSpec::new(
            vec![Expr::column("category"), Expr::column("class"), Expr::column("brand")],
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("detail_sales"), "sumsales")],
        )
        .with_grouping_sets(GroupingSetsSpec::new(
            vec![vec![false, false, false], vec![false, false, true], vec![false, true, true], vec![true, true, true]],
            "__grouping_id",
        )),
        Arc::clone(&aggregate_schema),
    );
    let window_input_sort = builder.add_sort(
        aggregate,
        SortSpec::new(vec![SortExpr::new(Expr::column("sumsales"), true, false)]),
        Arc::clone(&aggregate_schema),
    );
    let window = builder.add_window(
        window_input_sort,
        WindowSpec::new(
            vec![Expr::column("category")],
            vec![SortExpr::new(Expr::column("sumsales"), true, false)],
            vec![WindowFunction::rank("rk")],
        ),
        Arc::clone(&ranked_schema),
    );
    let filter = builder.add_filter(
        window,
        Expr::binary(Expr::column("rk"), BinaryOp::LtEq, Expr::literal(LiteralValue::UInt64(100))),
        Arc::clone(&ranked_schema),
    );
    let final_sort = builder.add_sort(
        filter,
        SortSpec::new(vec![
            SortExpr::new(Expr::column("category"), false, true),
            SortExpr::new(Expr::column("class"), false, true),
            SortExpr::new(Expr::column("brand"), false, true),
            SortExpr::new(Expr::column("sumsales"), true, false),
        ]),
        Arc::clone(&ranked_schema),
    );
    let limit = builder.add_limit(final_sort, LimitSpec::fetch(100), ranked_schema);
    let plan = builder.finish_with_host_sink(limit);
    let analysis = plan.capability_analysis();
    assert!(
        analysis.admits_rollup_rank_topk_after_aggregate_join_pushdown_pipeline(),
        "rollup/rank/top-k optimized shape should select the bounded capability envelope; first_rejection={:?}",
        analysis.first_rejection_projection()
    );

    let context = bounded_join_execution_context(&runtime);
    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.memory_control_policy.as_str(), "bounded");
    assert_eq!(metrics.hash_join_build_count, 1);
    assert_eq!(metrics.compiled_pipeline_attempts, 1);
    assert!(metrics.compiled_pipeline_tasks_executed >= 13);
    assert_eq!(metrics.compiled_pipeline_host_arrow_sinks, 1);
    assert_eq!(metrics.boundary.device_to_host_materializations, 1);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(batch.num_columns(), 6);
    assert_eq!(
        rollup_rank_rows(batch),
        vec![
            (None, None, None, 7, 53, 1),
            (Some("A".to_string()), None, None, 3, 35, 1),
            (Some("A".to_string()), Some("c1".to_string()), None, 1, 35, 1),
            (Some("A".to_string()), Some("c1".to_string()), Some("b1".to_string()), 0, 30, 3),
            (Some("A".to_string()), Some("c1".to_string()), Some("b2".to_string()), 0, 5, 8),
            (Some("B".to_string()), None, None, 3, 18, 1),
            (Some("B".to_string()), Some("c2".to_string()), None, 1, 18, 1),
            (Some("B".to_string()), Some("c2".to_string()), Some("b1".to_string()), 0, 7, 4),
            (Some("B".to_string()), Some("c2".to_string()), Some("b2".to_string()), 0, 11, 3),
        ],
        "rollup/rank/top-k output should preserve joined-id filtering, grouping-set rows, rank values, and final sort order",
    );

    fs::remove_dir_all(dir).unwrap();
}

fn rollup_rank_rows(batch: &RecordBatch) -> Vec<RollupRankRow> {
    let categories = batch.column(0).as_any().downcast_ref::<StringArray>().unwrap();
    let classes = batch.column(1).as_any().downcast_ref::<StringArray>().unwrap();
    let brands = batch.column(2).as_any().downcast_ref::<StringArray>().unwrap();
    let grouping_ids = batch.column(3).as_any().downcast_ref::<UInt32Array>().unwrap();
    let sales = int64_array(batch, 4);
    let ranks = uint64_array(batch, 5);
    (0..batch.num_rows())
        .map(|row| {
            (
                string_value(categories, row),
                string_value(classes, row),
                string_value(brands, row),
                grouping_ids.value(row),
                sales.value(row),
                ranks.value(row),
            )
        })
        .collect()
}

fn string_value(array: &StringArray, row: usize) -> Option<String> {
    (!array.is_null(row)).then(|| array.value(row).to_string())
}

fn write_rollup_rank_fact_parquet(path: &Path) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("category", DataType::Utf8, true),
        Field::new("class", DataType::Utf8, true),
        Field::new("brand", DataType::Utf8, true),
        Field::new("amount", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 1, 2, 2, 3, 4])),
            Arc::new(StringArray::from(vec![Some("A"), Some("A"), Some("A"), Some("B"), Some("B"), Some("C")])),
            Arc::new(StringArray::from(vec![Some("c1"), Some("c1"), Some("c1"), Some("c2"), Some("c2"), Some("c3")])),
            Arc::new(StringArray::from(vec![Some("b1"), Some("b1"), Some("b2"), Some("b1"), Some("b2"), Some("b3")])),
            Arc::new(Int64Array::from(vec![10, 20, 5, 7, 11, 13])),
        ],
    )
    .unwrap();
    write_single_batch_parquet(path, Arc::clone(&schema), batch);
    schema
}

fn write_rollup_rank_dim_parquet(path: &Path) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from(vec![1, 2, 3]))]).unwrap();
    write_single_batch_parquet(path, Arc::clone(&schema), batch);
    schema
}

fn write_single_batch_parquet(path: &Path, schema: Arc<Schema>, batch: RecordBatch) {
    let file = File::create(path).unwrap();
    let props = WriterProperties::builder().set_max_row_group_row_count(Some(3)).build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

fn bounded_stddev_join_aggregate_plan(runtime: &Arc<cudf::Runtime>, dir: &Path) -> QueryPlan {
    let left_path = dir.join("lineitem.parquet");
    let right_path = dir.join("orders.parquet");
    let left_schema = write_join_left_parquet(runtime, &left_path);
    let right_schema = write_join_right_parquet(runtime, &right_path);
    let join_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
    ]));
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty_stddev", DataType::Float64, true),
    ]));
    let sort = SortSpec::new(vec![SortExpr::new(Expr::column("qty_stddev"), true, true)]);

    let mut builder = QueryPlanBuilder::new();
    let left =
        builder.add_source(SourceNode::LocalParquet(join_left_source(&left_path, &left_schema, &["order_id", "qty"])));
    let right =
        builder.add_source(SourceNode::LocalParquet(join_right_source(&right_path, &right_schema, &["order_id"])));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(JoinType::Inner, vec![join_order_id_key()])
            .with_output_projection(vec![
                JoinOutputColumn::left(
                    nexus_query_engine::expr::ColumnRef::input(0, "order_id", "order_id"),
                    "order_id",
                ),
                JoinOutputColumn::left(nexus_query_engine::expr::ColumnRef::input(0, "qty", "qty"), "qty"),
            ])
            .with_build_side(JoinSide::Right),
        join_schema,
    );
    let aggregate = builder.add_aggregate(
        join,
        AggregateSpec::new(
            vec![Expr::column("order_id")],
            vec![AggregateExpr::new(AggregateFunction::StddevSamp, Expr::column("qty"), "qty_stddev")],
        ),
        Arc::clone(&aggregate_schema),
    );
    let sort = builder.add_sort(aggregate, sort, Arc::clone(&aggregate_schema));
    builder.finish_with_host_sink(sort)
}

#[test]
fn bounded_stddev_join_aggregate_uses_gpu_native_tolerant_statistical_contract() {
    let context = backend_context_with_grant_and_statistical_aggregate_mode(
        1024 * 1024,
        StatisticalAggregateExecutionMode::GpuNativeTolerant,
    );
    let runtime = Arc::new(context.runtime().unwrap());
    let dir = unique_temp_dir("nexus-query-engine-bounded-tolerant-stddev-join-aggregate");
    let plan = bounded_stddev_join_aggregate_plan(&runtime, &dir);

    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.compiled_pipeline_attempts, 1);
    assert!(metrics.aggregate_strategy_decisions.iter().any(|decision| {
        decision
            .facts
            .iter()
            .any(|(key, value)| key == "statistical_aggregate_execution_mode" && value == "gpu_native_tolerant")
            && decision
                .facts
                .iter()
                .any(|(key, value)| key == "statistical_aggregate_tolerance_contract" && value == "float_tolerant")
            && decision.facts.iter().any(|(key, value)| key == "statistical_aggregate_ddof" && value == "stddev_samp:1")
    }));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn bounded_stddev_join_aggregate_uses_exact_statistical_contract() {
    let context = backend_context_with_grant_and_statistical_aggregate_mode(
        1024 * 1024,
        StatisticalAggregateExecutionMode::DataFusionExactHostWelford,
    );
    let runtime = Arc::new(context.runtime().unwrap());
    let dir = unique_temp_dir("nexus-query-engine-bounded-exact-stddev-join-aggregate");
    let plan = bounded_stddev_join_aggregate_plan(&runtime, &dir);

    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.compiled_pipeline_attempts, 1);
    assert!(metrics.aggregate_strategy_decisions.iter().any(|decision| {
        decision.facts.iter().any(|(key, value)| {
            key == "statistical_aggregate_execution_mode" && value == "datafusion_exact_host_welford"
        })
    }));

    fs::remove_dir_all(dir).unwrap();
}
