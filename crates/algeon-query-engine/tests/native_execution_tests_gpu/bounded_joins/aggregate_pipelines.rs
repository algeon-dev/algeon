use std::{fs::File, path::Path, sync::Arc};

use query_planning::{expr::ColumnRef, mode::StatisticalAggregateExecutionMode};

use super::*;

type RollupRankRow = (Option<String>, Option<String>, Option<String>, u32, i64, u64);

#[test]
fn bounded_native_query_executes_root_hash_join_under_aggregate_with_ordered_reservations() {
    let runtime = backend_runtime();
    let dir = unique_temp_dir("query-engine-bounded-root-join-aggregate");
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
    let join = builder
        .add_join(
            left,
            right,
            JoinSpec::new(JoinType::Inner, vec![join_order_id_key()])
                .with_output_projection(vec![
                    JoinOutputColumn::left(ColumnRef::input(0, "order_id", "order_id"), "order_id"),
                    JoinOutputColumn::left(ColumnRef::input(0, "qty", "qty"), "qty"),
                ])
                .with_build_side(JoinSide::Right),
            join_schema,
        )
        .unwrap();
    let aggregate = builder
        .add_aggregate(
            join,
            AggregateSpec::new(
                vec![Expr::column("order_id")],
                vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("qty"), "line_count")],
            ),
            Arc::clone(&aggregate_schema),
        )
        .unwrap();
    let sort = builder.add_sort(aggregate, sort).unwrap();
    let limit = builder.add_limit(sort, LimitSpec::fetch(10)).unwrap();
    let plan = builder.finish_with_host_sink(limit).unwrap();
    let context = backend_context_with_grant(1024 * 1024);

    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
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
    let dir = unique_temp_dir("query-engine-bounded-compiled-sort-limit-topk");
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
    let join = builder
        .add_join(
            left,
            right,
            JoinSpec::new(JoinType::Inner, vec![join_order_id_key()])
                .with_output_projection(vec![
                    JoinOutputColumn::left(ColumnRef::input(0, "order_id", "order_id"), "order_id"),
                    JoinOutputColumn::left(ColumnRef::input(0, "qty", "qty"), "qty"),
                ])
                .with_build_side(JoinSide::Right),
            join_schema,
        )
        .unwrap();
    let aggregate = builder
        .add_aggregate(
            join,
            AggregateSpec::new(
                vec![Expr::column("order_id")],
                vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("qty"), "line_count")],
            ),
            Arc::clone(&aggregate_schema),
        )
        .unwrap();
    let sort = builder
        .add_sort(
            aggregate,
            SortSpec::new(vec![
                SortExpr::new(Expr::column("line_count"), true, false),
                SortExpr::new(Expr::column("order_id"), false, false),
            ]),
        )
        .unwrap();
    let limit = builder.add_limit(sort, LimitSpec::fetch(1)).unwrap();
    let plan = builder.finish_with_host_sink(limit).unwrap();
    let context = backend_context_with_grant(1024 * 1024);

    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert!(
        metrics.budgeted_execution_events.iter().any(|event| event.operation == "query_engine.final_sort.topk_prepare"
            && event.boundary == "final_sort_topk_stream_chunk"),
        "selective compiled Sort+Limit should top-k the lazily finalized aggregate output chunk by chunk"
    );
    assert!(
        metrics.budgeted_execution_events.iter().all(|event| event.operation != "query_engine.final_sort.prepare"),
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
    let dir = unique_temp_dir("query-engine-bounded-serial-aggregate-spine");
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
    let join = builder
        .add_join(
            customers,
            orders,
            JoinSpec::new(
                JoinType::Left,
                vec![JoinKey::new(
                    ColumnRef::input(0, "c_custkey", "c_custkey"),
                    ColumnRef::input(1, "o_custkey", "o_custkey"),
                )],
            )
            .with_output_projection(vec![
                JoinOutputColumn::left(ColumnRef::input(0, "c_custkey", "c_custkey"), "c_custkey"),
                JoinOutputColumn::right(ColumnRef::input(1, "o_orderkey", "o_orderkey"), "o_orderkey"),
            ])
            .with_build_side(JoinSide::Right),
            Arc::clone(&join_schema),
        )
        .unwrap();
    let projection = builder
        .add_projection(
            join,
            projection_outputs(vec![Expr::column("c_custkey"), Expr::column("o_orderkey")], Arc::clone(&join_schema)),
        )
        .unwrap();
    let customer_counts = builder
        .add_aggregate(
            projection,
            AggregateSpec::new(
                vec![Expr::column("c_custkey")],
                vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("o_orderkey"), "c_count")],
            ),
            customer_count_schema,
        )
        .unwrap();
    let count_projection = builder
        .add_projection(customer_counts, projection_outputs(vec![Expr::column("c_count")], c_count_schema))
        .unwrap();
    let distribution = builder
        .add_aggregate(
            count_projection,
            AggregateSpec::new(
                vec![Expr::column("c_count")],
                vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("c_count"), "custdist")],
            ),
            Arc::clone(&final_schema),
        )
        .unwrap();
    let sort = builder
        .add_sort(
            distribution,
            SortSpec::new(vec![
                SortExpr::new(Expr::column("custdist"), true, false),
                SortExpr::new(Expr::column("c_count"), true, false),
            ]),
        )
        .unwrap();
    let plan = builder.finish_with_host_sink(sort).unwrap();

    let context = bounded_join_execution_context(&runtime);
    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
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
    let dir = unique_temp_dir("query-engine-bounded-rollup-rank-topk");
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

    let mut builder = QueryPlanBuilder::new();
    let fact = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_path(
        &fact_path,
        fact_schema,
        vec!["id".into(), "category".into(), "class".into(), "brand".into(), "amount".into()],
    )));
    let detail_aggregate = builder
        .add_aggregate(
            fact,
            AggregateSpec::new(
                vec![Expr::column("id"), Expr::column("category"), Expr::column("class"), Expr::column("brand")],
                vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("amount"), "detail_sales")],
            ),
            Arc::clone(&detail_aggregate_schema),
        )
        .unwrap();
    let dim = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_path(
        &dim_path,
        dim_schema,
        vec!["id".into()],
    )));
    let join = builder
        .add_join(
            detail_aggregate,
            dim,
            JoinSpec::new(
                JoinType::Inner,
                vec![JoinKey::new(ColumnRef::input(0, "id", "id"), ColumnRef::input(1, "id", "id"))],
            )
            .with_output_projection(vec![
                JoinOutputColumn::left(ColumnRef::input(0, "category", "category"), "category"),
                JoinOutputColumn::left(ColumnRef::input(0, "class", "class"), "class"),
                JoinOutputColumn::left(ColumnRef::input(0, "brand", "brand"), "brand"),
                JoinOutputColumn::left(ColumnRef::input(0, "detail_sales", "detail_sales"), "detail_sales"),
            ])
            .with_build_side(JoinSide::Right),
            Arc::clone(&join_schema),
        )
        .unwrap();
    let aggregate = builder
        .add_aggregate(
            join,
            AggregateSpec::new(
                vec![Expr::column("category"), Expr::column("class"), Expr::column("brand")],
                vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("detail_sales"), "sumsales")],
            )
            .with_grouping_sets(GroupingSetsSpec::new(
                vec![
                    vec![false, false, false],
                    vec![false, false, true],
                    vec![false, true, true],
                    vec![true, true, true],
                ],
                "__grouping_id",
            )),
            Arc::clone(&aggregate_schema),
        )
        .unwrap();
    // DataFusion sorts a rank input on its partition key before the order
    // key, so each category's rows are adjacent when the rank counts them.
    let window_input_sort = builder
        .add_sort(
            aggregate,
            SortSpec::new(vec![
                SortExpr::new(Expr::column("category"), false, true),
                SortExpr::new(Expr::column("sumsales"), true, false),
            ]),
        )
        .unwrap();
    let window = builder
        .add_window(
            window_input_sort,
            WindowSpec::new(
                vec![Expr::column("category")],
                vec![SortExpr::new(Expr::column("sumsales"), true, false)],
                vec![WindowFunction::rank("rk")],
            ),
        )
        .unwrap();
    let filter = builder
        .add_filter(window, Expr::binary(Expr::column("rk"), BinaryOp::LtEq, Expr::literal(LiteralValue::UInt64(100))))
        .unwrap();
    let final_sort = builder
        .add_sort(
            filter,
            SortSpec::new(vec![
                SortExpr::new(Expr::column("category"), false, true),
                SortExpr::new(Expr::column("class"), false, true),
                SortExpr::new(Expr::column("brand"), false, true),
                SortExpr::new(Expr::column("sumsales"), true, false),
            ]),
        )
        .unwrap();
    let limit = builder.add_limit(final_sort, LimitSpec::fetch(100)).unwrap();
    let plan = builder.finish_with_host_sink(limit).unwrap();
    assert_bounded_pipeline_admitted(&plan);

    let context = bounded_join_execution_context(&runtime);
    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
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
            (Some("A".to_string()), Some("c1".to_string()), Some("b2".to_string()), 0, 5, 4),
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
    let join = builder
        .add_join(
            left,
            right,
            JoinSpec::new(JoinType::Inner, vec![join_order_id_key()])
                .with_output_projection(vec![
                    JoinOutputColumn::left(ColumnRef::input(0, "order_id", "order_id"), "order_id"),
                    JoinOutputColumn::left(ColumnRef::input(0, "qty", "qty"), "qty"),
                ])
                .with_build_side(JoinSide::Right),
            join_schema,
        )
        .unwrap();
    let aggregate = builder
        .add_aggregate(
            join,
            AggregateSpec::new(
                vec![Expr::column("order_id")],
                vec![AggregateExpr::new(AggregateFunction::StddevSamp, Expr::column("qty"), "qty_stddev")],
            ),
            Arc::clone(&aggregate_schema),
        )
        .unwrap();
    let sort = builder.add_sort(aggregate, sort).unwrap();
    builder.finish_with_host_sink(sort).unwrap()
}

#[test]
fn bounded_stddev_join_aggregate_uses_gpu_native_tolerant_statistical_contract() {
    let context = backend_context_with_grant_and_statistical_aggregate_mode(
        1024 * 1024,
        StatisticalAggregateExecutionMode::GpuNativeTolerant,
    );
    let runtime = Arc::new(context.runtime().unwrap());
    let dir = unique_temp_dir("query-engine-bounded-tolerant-stddev-join-aggregate");
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
    let dir = unique_temp_dir("query-engine-bounded-exact-stddev-join-aggregate");
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

/// Probe rows `(key, value = 1, d)` with `d` from `d_of_row`, written in row
/// groups of `row_group_rows`, joined on `d` against a dimension holding
/// `dimension_keys`, then grouped by `key` with `sum_count` SUMs of `value`.
fn grouped_sums_over_dimension_join_plan(
    dir: &Path,
    rows: usize,
    row_group_rows: usize,
    key_of_row: impl Fn(usize) -> i64,
    d_of_row: impl Fn(usize) -> i64,
    dimension_keys: &[i64],
    sum_count: usize,
) -> QueryPlan {
    let probe_path = dir.join("probe.parquet");
    let probe_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int64, false),
        Field::new("value", DataType::Int64, false),
        Field::new("d", DataType::Int64, false),
    ]));
    let probe = RecordBatch::try_new(
        Arc::clone(&probe_schema),
        vec![
            Arc::new(Int64Array::from((0..rows).map(&key_of_row).collect::<Vec<_>>())),
            Arc::new(Int64Array::from(vec![1_i64; rows])),
            Arc::new(Int64Array::from((0..rows).map(&d_of_row).collect::<Vec<_>>())),
        ],
    )
    .unwrap();
    write_record_batch_with_row_group_rows(&probe_path, Arc::clone(&probe_schema), probe, row_group_rows);
    let dimension_path = dir.join("dimension.parquet");
    let dimension_schema = Arc::new(Schema::new(vec![Field::new("d", DataType::Int64, false)]));
    let dimension =
        RecordBatch::try_new(Arc::clone(&dimension_schema), vec![Arc::new(Int64Array::from(dimension_keys.to_vec()))])
            .unwrap();
    write_record_batch_with_row_group_rows(&dimension_path, Arc::clone(&dimension_schema), dimension, 1024);

    let join_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int64, false),
        Field::new("value", DataType::Int64, false),
    ]));
    let mut aggregate_fields = vec![Field::new("key", DataType::Int64, false)];
    aggregate_fields.extend((0..sum_count).map(|index| Field::new(format!("sum_{index}"), DataType::Int64, true)));
    let aggregate = AggregateSpec::new(
        vec![Expr::column("key")],
        (0..sum_count)
            .map(|index| AggregateExpr::new(AggregateFunction::Sum, Expr::column("value"), format!("sum_{index}")))
            .collect(),
    );
    let mut builder = QueryPlanBuilder::new();
    let probe = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&probe_path).with_file_size_bytes(fs::metadata(&probe_path).unwrap().len())],
        probe_schema,
        vec!["key".to_string(), "value".to_string(), "d".to_string()],
    )));
    let dimension = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![LocalParquetFile::new(&dimension_path).with_file_size_bytes(fs::metadata(&dimension_path).unwrap().len())],
        dimension_schema,
        vec!["d".to_string()],
    )));
    let join = builder
        .add_join(
            probe,
            dimension,
            JoinSpec::new(
                JoinType::Inner,
                vec![JoinKey::new(ColumnRef::input(0, "d", "d"), ColumnRef::input(1, "d", "d"))],
            )
            .with_output_projection(vec![
                JoinOutputColumn::left(ColumnRef::input(0, "key", "key"), "key"),
                JoinOutputColumn::left(ColumnRef::input(0, "value", "value"), "value"),
            ])
            .with_build_side(JoinSide::Right),
            join_schema,
        )
        .unwrap();
    let aggregate = builder.add_aggregate(join, aggregate, Arc::new(Schema::new(aggregate_fields))).unwrap();
    let sort =
        builder.add_sort(aggregate, SortSpec::new(vec![SortExpr::new(Expr::column("key"), false, false)])).unwrap();
    builder.finish_with_host_sink(sort).unwrap()
}

/// Context whose 1 KiB source chunk cap reads every probe row group as its
/// own batch, so the aggregate receives one partial per row group.
fn row_group_batched_join_context() -> BackendQueryContext {
    backend_context_with_memory_control_policy(MemoryControlPolicy {
        device_budget_bytes: NATIVE_EXECUTION_TEST_QUERY_LIMIT_BYTES,
        max_source_chunk_bytes: 1024,
        max_row_groups_per_chunk: 1,
        max_retry_attempts: 3,
        min_retry_chunk_row_groups: 1,
    })
}

fn final_merge_fact<'a>(metrics: &'a QueryMetrics, fact: &str) -> Option<&'a str> {
    metrics
        .aggregate_strategy_decisions
        .iter()
        .find(|decision| decision.facts.contains_key("partitioned_merge_partition_count"))
        .and_then(|decision| decision.facts.get(fact))
        .map(String::as_str)
}

/// A probe chunk that joins to no rows still yields a zero-row partial. It
/// must not count as a second partial, which would replace the single
/// partial's direct projection with a partitioned final merge.
#[test]
fn bounded_join_aggregate_projects_a_single_partial_when_another_probe_chunk_joins_no_rows() {
    let dir = unique_temp_dir("query-engine-bounded-join-aggregate-empty-partial");
    let row_group_rows = 4096;
    // The first row group matches the dimension on every row. The second holds
    // only `d` values 1 and 3, so it joins to zero rows while its [1, 3] range
    // still overlaps the dimension's {0, 2} and is read.
    let plan = grouped_sums_over_dimension_join_plan(
        &dir,
        2 * row_group_rows,
        row_group_rows,
        |row| (row % 4) as i64,
        |row| if row < row_group_rows { 0 } else { 1 + 2 * (row % 2) as i64 },
        &[0, 2],
        1,
    );
    let context = row_group_batched_join_context();

    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.partial_aggregate_count, 2, "each probe batch yields a partial");
    assert_eq!(metrics.final_merge_attempts, 0);
    assert_eq!(final_merge_fact(metrics, "partitioned_merge_partition_count"), None);

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(int64_array(batch, 0).values().as_ref(), &[0, 1, 2, 3]);
    assert_eq!(int64_array(batch, 1).values().as_ref(), &[1024; 4]);

    fs::remove_dir_all(dir).unwrap();
}

/// The final merge sizes its partitions from the partial bytes it holds. The
/// join declares its 1,048,576 probe rows, but only 1,024 rows match and the
/// four partials hold 16 group rows (about 33 KB). Scaling those bytes by
/// declared probe rows over partial rows (65,536x, about 2 GiB) against the
/// 128 MiB aggregate partition target would select 17 partitions; the partial
/// bytes select one.
#[test]
fn bounded_join_aggregate_sizes_final_merge_partitions_from_partial_bytes_not_probe_rows() {
    let dir = unique_temp_dir("query-engine-bounded-join-aggregate-selective-partition-count");
    let rows = 1 << 20;
    let plan = grouped_sums_over_dimension_join_plan(
        &dir,
        rows,
        rows / 4,
        |row| ((row / 1024) % 4) as i64,
        |row| (row % 1024) as i64,
        &[0],
        256,
    );
    let context = row_group_batched_join_context();

    let result = execute_query_plan(&context, &plan).unwrap();
    let metrics = result.metrics();
    assert_eq!(metrics.partial_aggregate_count, 4);
    assert_eq!(metrics.final_merge_attempts, 1);
    assert_eq!(final_merge_fact(metrics, "partitioned_merge_incremental_partials"), Some("4"));
    assert_eq!(final_merge_fact(metrics, "partitioned_merge_partition_count"), Some("1"));

    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("host sink should materialize one RecordBatch");
    };
    assert_eq!(int64_array(batch, 0).values().as_ref(), &[0, 1, 2, 3]);
    for column in 1..=256 {
        assert_eq!(int64_array(batch, column).values().as_ref(), &[256; 4]);
    }

    fs::remove_dir_all(dir).unwrap();
}
