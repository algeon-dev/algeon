use std::{path::Path, sync::Arc};

use arrow::array::{Int32Array, Int64Array};
use query_engine::{dimension_pruning::local_dimension_pruning_resolver, rapids_backend::rapids_analysis_context};
use query_planning::{
    dynamic_filter::DynamicFilterConsumer,
    expr::ColumnRef,
    optimizer::{AggregateJoinPushdownRejectedReason, NativeOptimizerChange, optimize_query_plan},
    plan::QueryPlan,
};

use super::*;

#[test]
fn optimized_aggregate_join_pushdown_matches_plain_sum_inner_join_with_duplicate_keys_and_nulls() {
    let dir = unique_temp_dir("query-engine-aggregate-join-pushdown-parity");
    let left_path = dir.join("left.parquet");
    let right_path = dir.join("right.parquet");
    let context = backend_context();
    let runtime = Arc::new(context.runtime().unwrap());

    let left_schema = write_aggregate_join_pushdown_left_parquet(&runtime, &left_path);
    let right_schema = write_aggregate_join_pushdown_right_parquet(&runtime, &right_path);
    let plan = aggregate_join_pushdown_execution_plan(&left_path, &left_schema, &right_path, &right_schema);
    let optimizer_result = optimize_query_plan(
        plan.clone(),
        &Default::default(),
        rapids_analysis_context(PlanningPolicy::default()),
        local_dimension_pruning_resolver(),
    )
    .unwrap();
    assert_eq!(optimizer_result.change, NativeOptimizerChange::Changed);
    assert_eq!(optimizer_result.diagnostics.aggregate_join_pushdown.applied_count, 1);

    let original = execute_query_plan(&context, &plan).unwrap();
    let optimized = execute_query_plan(&context, &optimizer_result.plan).unwrap();

    let QueryOutput::HostArrow(original_batch) = original.output() else {
        panic!("original aggregate-join plan should materialize host output");
    };
    let QueryOutput::HostArrow(optimized_batch) = optimized.output() else {
        panic!("optimized aggregate-join plan should materialize host output");
    };
    assert_eq!(original_batch.schema(), optimized_batch.schema());
    assert_eq!(
        string_array(original_batch, 0).iter().collect::<Vec<_>>(),
        string_array(optimized_batch, 0).iter().collect::<Vec<_>>()
    );
    assert_eq!(
        int64_array(original_batch, 1).iter().collect::<Vec<_>>(),
        int64_array(optimized_batch, 1).iter().collect::<Vec<_>>()
    );
    assert_eq!(string_array(optimized_batch, 0).iter().collect::<Vec<_>>(), vec![Some("A"), Some("B")]);
    assert_eq!(int64_array(optimized_batch, 1).iter().collect::<Vec<_>>(), vec![Some(25), Some(14)]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn optimized_aggregate_join_pushdown_matches_left_join_count_with_unmatched_rows() {
    let dir = unique_temp_dir("query-engine-aggregate-join-pushdown-left-count-parity");
    let left_path = dir.join("customer.parquet");
    let right_path = dir.join("orders.parquet");
    let context = backend_context();
    let runtime = Arc::new(context.runtime().unwrap());

    let left_schema = write_left_count_customer_parquet(&runtime, &left_path);
    let right_schema = write_left_count_orders_parquet(&runtime, &right_path);
    let plan = aggregate_join_pushdown_left_count_plan(&left_path, &left_schema, &right_path, &right_schema);
    let optimizer_result = optimize_query_plan(
        plan.clone(),
        &Default::default(),
        rapids_analysis_context(PlanningPolicy::default()),
        local_dimension_pruning_resolver(),
    )
    .unwrap();
    assert_eq!(optimizer_result.change, NativeOptimizerChange::Changed);
    assert_eq!(optimizer_result.diagnostics.aggregate_join_pushdown.applied_count, 1);

    let original = execute_query_plan(&context, &plan).unwrap();
    let optimized = execute_query_plan(&context, &optimizer_result.plan).unwrap();

    let QueryOutput::HostArrow(original_batch) = original.output() else {
        panic!("original left count aggregate-join plan should materialize host output");
    };
    let QueryOutput::HostArrow(optimized_batch) = optimized.output() else {
        panic!("optimized left count aggregate-join plan should materialize host output");
    };
    assert_eq!(original_batch.schema(), optimized_batch.schema());
    assert_eq!(
        int32_array(original_batch, 0).iter().collect::<Vec<_>>(),
        int32_array(optimized_batch, 0).iter().collect::<Vec<_>>()
    );
    assert_eq!(
        int64_array(original_batch, 1).iter().collect::<Vec<_>>(),
        int64_array(optimized_batch, 1).iter().collect::<Vec<_>>()
    );
    assert_eq!(int32_array(optimized_batch, 0).iter().collect::<Vec<_>>(), vec![Some(1), Some(2), Some(3), Some(4)]);
    assert_eq!(int64_array(optimized_batch, 1).iter().collect::<Vec<_>>(), vec![Some(1), Some(4), Some(0), Some(0)]);

    fs::remove_dir_all(dir).unwrap();
}

fn aggregate_join_pushdown_execution_plan(
    left_path: &Path,
    left_schema: &Arc<Schema>,
    right_path: &Path,
    right_schema: &Arc<Schema>,
) -> QueryPlan {
    let join_schema = Arc::new(Schema::new(vec![
        Field::new("ss_net_profit", DataType::Int64, true),
        Field::new("s_store_name", DataType::Utf8, true),
    ]));
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("s_store_name", DataType::Utf8, true),
        Field::new("sum_profit", DataType::Int64, true),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(local_source(
        left_path,
        left_schema,
        &["zip_prefix", "s_store_name", "ss_net_profit"],
    )));
    let right = builder.add_source(SourceNode::LocalParquet(local_source(right_path, right_schema, &["zip_prefix"])));
    let join = builder
        .add_join(
            left,
            right,
            JoinSpec::new(
                JoinType::Inner,
                vec![JoinKey::new(
                    ColumnRef::input(0, "zip_prefix", "zip_prefix"),
                    ColumnRef::input(1, "zip_prefix", "zip_prefix"),
                )],
            )
            .with_output_projection(vec![
                JoinOutputColumn::left(ColumnRef::input(0, "ss_net_profit", "ss_net_profit"), "ss_net_profit"),
                JoinOutputColumn::left(ColumnRef::input(0, "s_store_name", "s_store_name"), "s_store_name"),
            ])
            .with_build_side(JoinSide::Right),
            join_schema,
        )
        .unwrap();
    let projection = builder
        .add_projection(
            join,
            vec![
                ProjectionOutput::new(Expr::column("s_store_name"), Field::new("s_store_name", DataType::Utf8, true)),
                ProjectionOutput::new(
                    Expr::column("ss_net_profit"),
                    Field::new("__algeon_native_aggregate_arg_0", DataType::Int64, true),
                ),
            ],
        )
        .unwrap();
    let aggregate = builder
        .add_aggregate(
            projection,
            AggregateSpec::new(
                vec![Expr::column("s_store_name")],
                vec![AggregateExpr::new(
                    AggregateFunction::Sum,
                    Expr::column("__algeon_native_aggregate_arg_0"),
                    "sum_profit",
                )],
            ),
            Arc::clone(&aggregate_schema),
        )
        .unwrap();
    let sort = builder
        .add_sort(aggregate, SortSpec::new(vec![SortExpr::new(Expr::column("s_store_name"), false, false)]))
        .unwrap();
    builder.finish_with_host_sink(sort).unwrap()
}

fn aggregate_join_pushdown_left_count_plan(
    left_path: &Path,
    left_schema: &Arc<Schema>,
    right_path: &Path,
    right_schema: &Arc<Schema>,
) -> QueryPlan {
    let join_schema = Arc::new(Schema::new(vec![
        Field::new("c_custkey", DataType::Int32, false),
        Field::new("o_orderkey", DataType::Int64, true),
    ]));
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("c_custkey", DataType::Int32, false),
        Field::new("c_count", DataType::Int64, false),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(local_source(left_path, left_schema, &["c_custkey"])));
    let right = builder.add_source(SourceNode::LocalParquet(local_source(
        right_path,
        right_schema,
        &["o_custkey", "o_orderkey"],
    )));
    let join = builder
        .add_join(
            left,
            right,
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
            join_schema,
        )
        .unwrap();
    let aggregate = builder
        .add_aggregate(
            join,
            AggregateSpec::new(
                vec![Expr::column("c_custkey")],
                vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("o_orderkey"), "c_count")],
            ),
            Arc::clone(&aggregate_schema),
        )
        .unwrap();
    let sort = builder
        .add_sort(aggregate, SortSpec::new(vec![SortExpr::new(Expr::column("c_custkey"), false, true)]))
        .unwrap();
    builder.finish_with_host_sink(sort).unwrap()
}

fn local_source(path: &Path, schema: &Arc<Schema>, columns: &[&str]) -> NativeLocalParquetSource {
    LocalParquetSource::from_files(
        vec![LocalParquetFile::new(path).with_file_size_bytes(fs::metadata(path).unwrap().len())],
        Arc::clone(schema),
        columns.iter().map(|column| (*column).to_string()).collect(),
    )
}

fn write_aggregate_join_pushdown_left_parquet(runtime: &cudf::Runtime, path: &Path) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("zip_prefix", DataType::Int32, true),
        Field::new("s_store_name", DataType::Utf8, true),
        Field::new("ss_net_profit", DataType::Int64, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![Some(1), Some(1), Some(2), Some(3), Some(3), None])),
            Arc::new(StringArray::from(vec![Some("A"), Some("A"), Some("A"), Some("B"), Some("B"), Some("B")])),
            Arc::new(Int64Array::from(vec![Some(10), None, Some(5), Some(7), None, Some(100)])),
        ],
    )
    .unwrap();
    DataFrame::from_record_batch(runtime, &batch).unwrap().write_parquet(path).unwrap();
    schema
}

fn write_aggregate_join_pushdown_right_parquet(runtime: &cudf::Runtime, path: &Path) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![Field::new("zip_prefix", DataType::Int32, true)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(vec![Some(1), Some(1), Some(2), Some(3), Some(3), None]))],
    )
    .unwrap();
    DataFrame::from_record_batch(runtime, &batch).unwrap().write_parquet(path).unwrap();
    schema
}

fn write_left_count_customer_parquet(runtime: &cudf::Runtime, path: &Path) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![Field::new("c_custkey", DataType::Int32, false)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(vec![Some(1), Some(2), Some(2), Some(3), Some(4)]))],
    )
    .unwrap();
    DataFrame::from_record_batch(runtime, &batch).unwrap().write_parquet(path).unwrap();
    schema
}

fn write_left_count_orders_parquet(runtime: &cudf::Runtime, path: &Path) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("o_custkey", DataType::Int32, false),
        Field::new("o_orderkey", DataType::Int64, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![Some(1), Some(1), Some(2), Some(2), Some(2), Some(5)])),
            Arc::new(Int64Array::from(vec![Some(10), None, Some(20), Some(21), None, Some(50)])),
        ],
    )
    .unwrap();
    DataFrame::from_record_batch(runtime, &batch).unwrap().write_parquet(path).unwrap();
    schema
}

const SEMI_GROUP_FILTER_ROWS: i32 = 2_000_000;
const SEMI_GROUP_FILTER_ROW_GROUP_ROWS: usize = 125_000;
// Between the semi-first peak (chunks plus a handful of groups) and the peak of
// pre-aggregating every detail row before the keyset filter; see the plan doc
// for the measured values.
const SEMI_GROUP_FILTER_GRANT_BYTES: u64 = 24 * 1024 * 1024;

// One group per detail row (null keys every 100_000th row) so a pre-aggregate
// below the semi filter would retain all two million groups.
fn write_semi_group_filter_detail_parquet(path: &Path) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, true),
        Field::new("qty", DataType::Int64, false),
    ]));
    let order_ids = (0..SEMI_GROUP_FILTER_ROWS).map(|row| (row % 100_000 != 0).then_some(row)).collect::<Vec<_>>();
    let quantities = (0..SEMI_GROUP_FILTER_ROWS).map(|row| i64::from(row) * 2 + 1).collect::<Vec<_>>();
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(order_ids)), Arc::new(Int64Array::from(quantities))],
    )
    .unwrap();
    write_record_batch_with_row_group_rows(path, Arc::clone(&schema), batch, SEMI_GROUP_FILTER_ROW_GROUP_ROWS);
    schema
}

fn write_semi_group_filter_keyset_parquet(path: &Path, order_ids: &[i32]) -> Arc<Schema> {
    let schema = Arc::new(Schema::new(vec![Field::new("order_id", DataType::Int32, false)]));
    let batch =
        RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from(order_ids.to_vec()))]).unwrap();
    write_record_batch_with_row_group_rows(path, Arc::clone(&schema), batch, SEMI_GROUP_FILTER_ROW_GROUP_ROWS);
    schema
}

fn left_semi_group_filter_aggregate_plan(
    detail_path: &Path,
    detail_schema: &Arc<Schema>,
    keyset_path: &Path,
    keyset_schema: &Arc<Schema>,
    keyset_filter: Option<Expr>,
) -> QueryPlan {
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, true),
        Field::new("sum_qty", DataType::Int64, true),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let detail = builder.add_source(SourceNode::LocalParquet(join_left_source(
        detail_path,
        detail_schema,
        &["order_id", "qty"],
    )));
    let keyset =
        builder.add_source(SourceNode::LocalParquet(join_right_source(keyset_path, keyset_schema, &["order_id"])));
    let keyset = match keyset_filter {
        Some(predicate) => builder.add_filter(keyset, predicate).unwrap(),
        None => keyset,
    };
    let semi = builder
        .add_join(
            detail,
            keyset,
            JoinSpec::new(
                JoinType::LeftSemi,
                vec![JoinKey::new(
                    ColumnRef::input(0, "order_id", "order_id"),
                    ColumnRef::input(1, "order_id", "order_id"),
                )],
            )
            .with_output_projection(vec![
                JoinOutputColumn::left(ColumnRef::input(0, "order_id", "order_id"), "order_id"),
                JoinOutputColumn::left(ColumnRef::input(0, "qty", "qty"), "qty"),
            ])
            .with_build_side(JoinSide::Right),
            Arc::clone(detail_schema),
        )
        .unwrap();
    let aggregate = builder
        .add_aggregate(
            semi,
            AggregateSpec::new(
                vec![Expr::column("order_id")],
                vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("qty"), "sum_qty")],
            ),
            Arc::clone(&aggregate_schema),
        )
        .unwrap();
    let sort = builder
        .add_sort(aggregate, SortSpec::new(vec![SortExpr::new(Expr::column("order_id"), false, false)]))
        .unwrap();
    builder.finish_with_host_sink(sort).unwrap()
}

#[test]
fn bounded_native_query_filters_left_semi_groups_before_the_final_aggregate() {
    let dir = unique_temp_dir("query-engine-left-semi-group-filter-before-aggregate");
    let detail_path = dir.join("detail.parquet");
    let keyset_path = dir.join("keyset.parquet");
    let detail_schema = write_semi_group_filter_detail_parquet(&detail_path);
    // Duplicate keys, a key no detail row carries, and a key that only null
    // detail rows would match under null-equal semantics.
    let keyset_schema =
        write_semi_group_filter_keyset_parquet(&keyset_path, &[1_500_003, 7, 7, SEMI_GROUP_FILTER_ROWS + 5, 42]);

    let plan = left_semi_group_filter_aggregate_plan(&detail_path, &detail_schema, &keyset_path, &keyset_schema, None);
    let optimizer_result = optimize_query_plan(
        plan.clone(),
        &Default::default(),
        rapids_analysis_context(PlanningPolicy::default()),
        local_dimension_pruning_resolver(),
    )
    .unwrap();
    let pushdown = &optimizer_result.diagnostics.aggregate_join_pushdown;
    assert_eq!(pushdown.applied_count, 0, "{pushdown:?}");
    assert_eq!(
        pushdown.candidates[0].rejected_reason,
        Some(AggregateJoinPushdownRejectedReason::LeftSemiAlreadyFiltersFinalGroups)
    );
    assert_eq!(
        (optimizer_result.plan.nodes(), optimizer_result.plan.sink()),
        (plan.nodes(), plan.sink()),
        "the semi filter must stay directly below the grouped aggregate"
    );
    let routes = optimizer_result.plan.dynamic_filter_routes();
    assert_eq!(routes.len(), 1, "the semi join build keys should reduce the detail source: {routes:?}");
    let detail_source = plan.nodes()[0].id;
    assert_eq!(routes[0].consumer, DynamicFilterConsumer::LocalParquetSource { source_node_id: detail_source });

    let context = backend_context_with_grant(SEMI_GROUP_FILTER_GRANT_BYTES);
    let (result, _metrics, peak_bytes) = execute_admitted_with_peak(&context, &optimizer_result.plan);
    let output = result.unwrap_or_else(|error| panic!("semi-first grouped aggregate should fit the grant: {error:?}"));
    let QueryOutput::HostArrow(batch) = &output else {
        panic!("grouped aggregate should materialize host output");
    };
    assert_eq!(int32_array(batch, 0).iter().collect::<Vec<_>>(), vec![Some(7), Some(42), Some(1_500_003)]);
    assert_eq!(int64_array(batch, 1).iter().collect::<Vec<_>>(), vec![Some(15), Some(85), Some(3_000_007)]);
    assert!(
        peak_bytes < SEMI_GROUP_FILTER_GRANT_BYTES / 2,
        "semi-first execution should stay far below the grant: peak={peak_bytes}"
    );

    // A keyset predicate that no key satisfies (Q18's `sum > 300` on a small
    // dataset) leaves an empty build side.
    let empty_plan = left_semi_group_filter_aggregate_plan(
        &detail_path,
        &detail_schema,
        &keyset_path,
        &keyset_schema,
        Some(Expr::binary(Expr::column("order_id"), BinaryOp::Lt, Expr::literal(LiteralValue::Int32(0)))),
    );
    let (result, _metrics, _peak_bytes) = execute_admitted_with_peak(&context, &empty_plan);
    let QueryOutput::HostArrow(batch) = result.unwrap() else {
        panic!("grouped aggregate should materialize host output");
    };
    assert_eq!(batch.num_rows(), 0, "an empty keyset filters every group");

    fs::remove_dir_all(dir).unwrap();
}
