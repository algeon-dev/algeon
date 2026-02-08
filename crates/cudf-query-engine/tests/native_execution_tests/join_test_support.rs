use super::*;

#[path = "join_test_support/mixed_aggregate.rs"]
mod mixed_aggregate;

pub(super) use mixed_aggregate::*;

pub(super) fn join_left_source(path: &Path, schema: &Arc<Schema>, columns: &[&str]) -> LocalParquetSource {
    LocalParquetSource::from_files(
        vec![LocalParquetFile::new(path).with_file_size_bytes(fs::metadata(path).unwrap().len())],
        Arc::clone(schema),
        columns.iter().map(|column| (*column).to_string()).collect(),
    )
}

pub(super) fn join_right_source(path: &Path, schema: &Arc<Schema>, columns: &[&str]) -> LocalParquetSource {
    LocalParquetSource::from_files(
        vec![LocalParquetFile::new(path).with_file_size_bytes(fs::metadata(path).unwrap().len())],
        Arc::clone(schema),
        columns.iter().map(|column| (*column).to_string()).collect(),
    )
}

pub(super) fn join_order_id_key() -> JoinKey {
    JoinKey::new(
        cudf_query_engine::ColumnRef::input(0, "order_id", "left_order_id"),
        cudf_query_engine::ColumnRef::input(1, "order_id", "right_order_id"),
    )
}

pub(super) fn id_join_key() -> JoinKey {
    JoinKey::new(
        cudf_query_engine::ColumnRef::input(0, "id", "left_id"),
        cudf_query_engine::ColumnRef::input(1, "id", "right_id"),
    )
}

pub(super) fn right_order_id_times_ten() -> Expr {
    Expr::binary(
        Expr::input_column(1, "order_id", "right_order_id"),
        BinaryOp::Multiply,
        Expr::literal(LiteralValue::Int32(10)),
    )
}

pub(super) fn qty_gt_right_order_id_times_ten() -> Expr {
    Expr::binary(Expr::input_column(0, "qty", "qty"), BinaryOp::Gt, right_order_id_times_ten())
}

pub(super) fn bounded_join_execution_context(runtime: &Arc<cudf::Runtime>) -> ExecutionContext {
    ExecutionContext::builder()
        .with_shared_runtime(Arc::clone(runtime))
        .with_memory_control_policy(MemoryControlPolicy::Bounded {
            device_budget_bytes: 1024 * 1024,
            max_source_chunk_bytes: 1024,
            max_row_groups_per_chunk: 1,
            max_retry_attempts: 1,
            min_retry_chunk_row_groups: 1,
        })
        .build()
        .unwrap()
}

pub(super) fn execute_bounded_keyless_left_only_conditional_join(
    runtime: &Arc<cudf::Runtime>,
    left_path: &Path,
    right_path: &Path,
    left_schema: &Arc<Schema>,
    right_schema: &Arc<Schema>,
    join_type: JoinType,
    build_side: JoinSide,
) -> cudf_query_engine::QueryExecutionResult {
    let output_schema = Arc::new(Schema::new(vec![Field::new("qty", DataType::Int32, false)]));

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(join_left_source(left_path, left_schema, &["qty"])));
    let right =
        builder.add_source(SourceNode::LocalParquet(join_right_source(right_path, right_schema, &["order_id"])));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(join_type, vec![])
            .with_residual_predicate(qty_gt_right_order_id_times_ten())
            .with_output_projection(vec![JoinOutputColumn::left(
                cudf_query_engine::ColumnRef::input(0, "qty", "qty"),
                "qty",
            )])
            .with_build_side(build_side),
        Arc::clone(&output_schema),
    );
    let plan = builder.finish_with_host_sink(join);
    assert!(plan.capability_analysis().admits_root_join());
    let context = bounded_join_execution_context(runtime);
    execute_query_plan(&context, &plan).unwrap()
}

pub(super) fn execute_bounded_equality_left_only_filtered_join(
    runtime: &Arc<cudf::Runtime>,
    left_path: &Path,
    right_path: &Path,
    left_schema: &Arc<Schema>,
    right_schema: &Arc<Schema>,
    join_type: JoinType,
) -> cudf_query_engine::QueryExecutionResult {
    let output_schema = Arc::new(Schema::new(vec![Field::new("qty", DataType::Int32, false)]));

    let mut builder = QueryPlanBuilder::new();
    let left =
        builder.add_source(SourceNode::LocalParquet(join_left_source(left_path, left_schema, &["order_id", "qty"])));
    let right =
        builder.add_source(SourceNode::LocalParquet(join_right_source(right_path, right_schema, &["order_id"])));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(join_type, vec![join_order_id_key()])
            .with_output_projection(vec![JoinOutputColumn::left(
                cudf_query_engine::ColumnRef::input(0, "qty", "qty"),
                "qty",
            )])
            .with_build_side(JoinSide::Right),
        Arc::clone(&output_schema),
    );
    let plan = builder.finish_with_host_sink(join);
    assert!(plan.capability_analysis().admits_root_join());
    let context = bounded_join_execution_context(runtime);
    execute_query_plan(&context, &plan).unwrap()
}

#[allow(clippy::too_many_arguments)]
pub(super) fn keyless_conditional_join_input_subplan_plan(
    lineitem_path: &Path,
    anti_path: &Path,
    threshold_path: &Path,
    lineitem_schema: &Arc<Schema>,
    anti_schema: &Arc<Schema>,
    threshold_schema: &Arc<Schema>,
    left_subplan_schema: Arc<Schema>,
    threshold_aggregate_schema: Arc<Schema>,
    output_schema: Arc<Schema>,
    gpu_sink: bool,
) -> cudf_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let lineitem = builder.add_source(SourceNode::LocalParquet(join_left_source(
        lineitem_path,
        lineitem_schema,
        &["order_id", "qty"],
    )));
    let anti_orders =
        builder.add_source(SourceNode::LocalParquet(join_right_source(anti_path, anti_schema, &["order_id"])));
    let threshold_source = builder.add_source(SourceNode::LocalParquet(join_right_source(
        threshold_path,
        threshold_schema,
        &["order_id"],
    )));
    let anti_join = builder.add_join(
        lineitem,
        anti_orders,
        JoinSpec::new(JoinType::LeftAnti, vec![join_order_id_key()])
            .with_output_projection(vec![
                JoinOutputColumn::left(cudf_query_engine::ColumnRef::input(0, "order_id", "order_id"), "order_id"),
                JoinOutputColumn::left(cudf_query_engine::ColumnRef::input(0, "qty", "qty"), "qty"),
            ])
            .with_build_side(JoinSide::Right),
        Arc::clone(&left_subplan_schema),
    );
    let anti_projection = builder.add_projection(
        anti_join,
        projection_outputs(vec![Expr::column("order_id"), Expr::column("qty")], Arc::clone(&left_subplan_schema)),
    );
    let threshold = builder.add_aggregate(
        threshold_source,
        AggregateSpec::new(
            Vec::new(),
            vec![AggregateExpr::new(AggregateFunction::Max, Expr::column("order_id"), "threshold")],
        ),
        threshold_aggregate_schema,
    );
    let root = builder.add_join(
        anti_projection,
        threshold,
        JoinSpec::new(JoinType::Inner, Vec::new())
            .with_residual_predicate(Expr::binary(
                Expr::input_column(0, "qty", "qty"),
                BinaryOp::Gt,
                Expr::input_column(1, "threshold", "threshold"),
            ))
            .with_output_projection(vec![
                JoinOutputColumn::left(cudf_query_engine::ColumnRef::input(0, "order_id", "order_id"), "order_id"),
                JoinOutputColumn::left(cudf_query_engine::ColumnRef::input(0, "qty", "qty"), "qty"),
                JoinOutputColumn::right(cudf_query_engine::ColumnRef::input(1, "threshold", "threshold"), "threshold"),
            ])
            .with_build_side(JoinSide::Left),
        Arc::clone(&output_schema),
    );
    let sort = builder.add_sort(
        root,
        SortSpec::new(vec![
            SortExpr::new(Expr::column("order_id"), false, false),
            SortExpr::new(Expr::column("qty"), false, false),
        ]),
        output_schema,
    );
    if gpu_sink { builder.finish_gpu_resident(sort) } else { builder.finish_with_host_sink(sort) }
}

pub(super) fn keyless_conditional_aggregate_over_join_subplan_inputs_plan(
    grouped_fact_path: &Path,
    scalar_fact_path: &Path,
    supplier_path: &Path,
    grouped_fact_schema: &Arc<Schema>,
    scalar_fact_schema: &Arc<Schema>,
    supplier_schema: &Arc<Schema>,
) -> cudf_query_engine::QueryPlan {
    let join_schema = Arc::new(Schema::new(vec![
        Field::new("ps_partkey", DataType::Int32, false),
        Field::new("qty", DataType::Int64, false),
    ]));
    let scalar_join_schema = Arc::new(Schema::new(vec![Field::new("qty", DataType::Int64, false)]));
    let grouped_aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("ps_partkey", DataType::Int32, false),
        Field::new("value", DataType::Int64, true),
    ]));
    let scalar_aggregate_schema = Arc::new(Schema::new(vec![Field::new("threshold", DataType::Int64, true)]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("ps_partkey", DataType::Int32, false),
        Field::new("value", DataType::Int64, true),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let grouped_partsupp = builder
        .add_source(SourceNode::LocalParquet(supplier_cost_partsupp_source(grouped_fact_path, grouped_fact_schema)));
    let grouped_supplier =
        builder.add_source(SourceNode::LocalParquet(supplier_cost_supplier_source(supplier_path, supplier_schema)));
    let grouped_join = builder.add_join(
        grouped_partsupp,
        grouped_supplier,
        JoinSpec::new(JoinType::Inner, vec![supplier_cost_suppkey_join_key("ps_suppkey", "s_suppkey")])
            .with_output_projection(vec![
                JoinOutputColumn::left(
                    cudf_query_engine::ColumnRef::input(0, "ps_partkey", "ps_partkey"),
                    "ps_partkey",
                ),
                JoinOutputColumn::left(cudf_query_engine::ColumnRef::input(0, "qty", "qty"), "qty"),
            ])
            .with_build_side(JoinSide::Right),
        Arc::clone(&join_schema),
    );
    let grouped_aggregate = builder.add_aggregate(
        grouped_join,
        AggregateSpec::new(
            vec![Expr::column("ps_partkey")],
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("qty"), "value")],
        ),
        Arc::clone(&grouped_aggregate_schema),
    );
    let grouped_projection = builder.add_projection(
        grouped_aggregate,
        projection_outputs(
            vec![Expr::column("ps_partkey"), Expr::column("value")],
            Arc::clone(&grouped_aggregate_schema),
        ),
    );

    let scalar_partsupp = builder
        .add_source(SourceNode::LocalParquet(supplier_cost_partsupp_source(scalar_fact_path, scalar_fact_schema)));
    let scalar_supplier =
        builder.add_source(SourceNode::LocalParquet(supplier_cost_supplier_source(supplier_path, supplier_schema)));
    let scalar_join = builder.add_join(
        scalar_partsupp,
        scalar_supplier,
        JoinSpec::new(JoinType::Inner, vec![supplier_cost_suppkey_join_key("ps_suppkey", "s_suppkey")])
            .with_output_projection(vec![JoinOutputColumn::left(
                cudf_query_engine::ColumnRef::input(0, "qty", "qty"),
                "qty",
            )])
            .with_build_side(JoinSide::Right),
        scalar_join_schema,
    );
    let scalar_aggregate = builder.add_aggregate(
        scalar_join,
        AggregateSpec::new(
            Vec::new(),
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("qty"), "threshold")],
        ),
        Arc::clone(&scalar_aggregate_schema),
    );
    let scalar_projection = builder
        .add_projection(scalar_aggregate, projection_outputs(vec![Expr::column("threshold")], scalar_aggregate_schema));

    let root = builder.add_join(
        grouped_projection,
        scalar_projection,
        JoinSpec::new(JoinType::Inner, Vec::new())
            .with_residual_predicate(Expr::binary(
                Expr::input_column(0, "value", "value"),
                BinaryOp::Gt,
                Expr::input_column(1, "threshold", "threshold"),
            ))
            .with_output_projection(vec![
                JoinOutputColumn::left(
                    cudf_query_engine::ColumnRef::input(0, "ps_partkey", "ps_partkey"),
                    "ps_partkey",
                ),
                JoinOutputColumn::left(cudf_query_engine::ColumnRef::input(0, "value", "value"), "value"),
            ])
            .with_build_side(JoinSide::Left),
        Arc::clone(&output_schema),
    );
    let final_projection = builder.add_projection(
        root,
        projection_outputs(vec![Expr::column("ps_partkey"), Expr::column("value")], Arc::clone(&output_schema)),
    );
    let sort = builder.add_sort(
        final_projection,
        SortSpec::new(vec![
            SortExpr::new(Expr::column("value"), true, false),
            SortExpr::new(Expr::column("ps_partkey"), false, false),
        ]),
        output_schema,
    );
    builder.finish_with_host_sink(sort)
}

pub(super) fn supplier_cost_partsupp_source(path: &Path, schema: &Arc<Schema>) -> LocalParquetSource {
    LocalParquetSource::from_files(
        vec![LocalParquetFile::new(path).with_file_size_bytes(fs::metadata(path).unwrap().len())],
        Arc::clone(schema),
        vec!["ps_partkey".to_string(), "ps_suppkey".to_string(), "qty".to_string()],
    )
}

pub(super) fn supplier_cost_supplier_source(path: &Path, schema: &Arc<Schema>) -> LocalParquetSource {
    LocalParquetSource::from_files(
        vec![LocalParquetFile::new(path).with_file_size_bytes(fs::metadata(path).unwrap().len())],
        Arc::clone(schema),
        vec!["s_suppkey".to_string()],
    )
}

pub(super) fn supplier_cost_suppkey_join_key(left_name: &str, right_name: &str) -> JoinKey {
    JoinKey::new(
        cudf_query_engine::ColumnRef::input(0, left_name, left_name),
        cudf_query_engine::ColumnRef::input(1, right_name, right_name),
    )
}

pub(super) fn assert_empty_bounded_filtered_join_result(
    result: &cudf_query_engine::QueryExecutionResult,
    join_type: &str,
) {
    let QueryOutput::HostArrow(batch) = result.output() else {
        panic!("empty bounded filtered join should return host output");
    };
    assert_eq!(batch.num_rows(), 0);
    assert_eq!(batch.num_columns(), 1);

    let metrics = result.metrics();
    let join_metrics =
        metrics.operators.iter().find_map(|operator| operator.join.as_ref()).expect("join metrics should be recorded");
    assert_eq!(join_metrics.join_type, join_type);
    assert_eq!(join_metrics.execution_strategy, "bounded_reusable_filtered_join");
    assert_eq!(join_metrics.output_rows, 0);
    assert_eq!(join_metrics.gather_map_bytes, 0);
    assert!(join_metrics.hash_table_bytes > 0);
    assert!(join_metrics.filtered_probe_state_bytes > 0);
    assert_eq!(metrics.hash_join_build_count, 0);
    assert_eq!(metrics.hash_join_hash_table_bytes, 0);
    assert_eq!(metrics.filtered_join_build_count, 1);
    assert_eq!(metrics.filtered_join_hash_table_bytes, join_metrics.hash_table_bytes);
    assert_eq!(metrics.filtered_join_probe_state_bytes, join_metrics.filtered_probe_state_bytes);
}

pub(super) fn execute_left_only_residual_join(
    runtime: &Arc<cudf::Runtime>,
    left_path: &Path,
    right_path: &Path,
    left_schema: &Arc<Schema>,
    right_schema: &Arc<Schema>,
    join_type: JoinType,
) -> cudf_query_engine::QueryExecutionResult {
    let output_schema = Arc::new(Schema::new(vec![Field::new("qty", DataType::Int32, false)]));

    let mut builder = QueryPlanBuilder::new();
    let left =
        builder.add_source(SourceNode::LocalParquet(join_left_source(left_path, left_schema, &["order_id", "qty"])));
    let right =
        builder.add_source(SourceNode::LocalParquet(join_right_source(right_path, right_schema, &["order_id"])));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(join_type, vec![join_order_id_key()])
            .with_residual_predicate(qty_gt_right_order_id_times_ten())
            .with_output_projection(vec![JoinOutputColumn::left(
                cudf_query_engine::ColumnRef::input(0, "qty", "qty"),
                "qty",
            )])
            .with_build_side(JoinSide::Right),
        Arc::clone(&output_schema),
    );
    let sort = builder.add_sort(
        join,
        SortSpec::new(vec![SortExpr::new(Expr::column("qty"), false, false)]),
        Arc::clone(&output_schema),
    );
    let plan = builder.finish_with_host_sink(sort);
    let context = ExecutionContext::from_shared_runtime(Arc::clone(runtime));
    execute_query_plan(&context, &plan).unwrap()
}

pub(super) fn execute_bounded_left_only_residual_join(
    runtime: &Arc<cudf::Runtime>,
    left_path: &Path,
    right_path: &Path,
    left_schema: &Arc<Schema>,
    right_schema: &Arc<Schema>,
    join_type: JoinType,
) -> cudf_query_engine::QueryExecutionResult {
    let output_schema = Arc::new(Schema::new(vec![Field::new("qty", DataType::Int32, false)]));

    let mut builder = QueryPlanBuilder::new();
    let left =
        builder.add_source(SourceNode::LocalParquet(join_left_source(left_path, left_schema, &["order_id", "qty"])));
    let right =
        builder.add_source(SourceNode::LocalParquet(join_right_source(right_path, right_schema, &["order_id"])));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(join_type, vec![join_order_id_key()])
            .with_residual_predicate(qty_gt_right_order_id_times_ten())
            .with_output_projection(vec![JoinOutputColumn::left(
                cudf_query_engine::ColumnRef::input(0, "qty", "qty"),
                "qty",
            )])
            .with_build_side(JoinSide::Right),
        Arc::clone(&output_schema),
    );
    let plan = builder.finish_with_host_sink(join);
    assert!(plan.capability_analysis().admits_root_join());
    let context = bounded_join_execution_context(runtime);
    execute_query_plan(&context, &plan).unwrap()
}

pub(super) fn residual_validation_error(
    join_type: JoinType,
    keys: Vec<JoinKey>,
    residual: Expr,
) -> cudf_query_engine::QueryError {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("left_id", DataType::Int32, true),
        Field::new("right_id", DataType::Int32, true),
    ]));
    let left_source = LocalParquetSource::new(
        "/tmp/does-not-need-to-exist-for-residual-validation-left.parquet",
        Arc::clone(&schema),
        vec!["id".into()],
    );
    let right_source = LocalParquetSource::new(
        "/tmp/does-not-need-to-exist-for-residual-validation-right.parquet",
        Arc::clone(&schema),
        vec!["id".into()],
    );

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(left_source));
    let right = builder.add_source(SourceNode::LocalParquet(right_source));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(join_type, keys).with_residual_predicate(residual).with_output_projection(vec![
            JoinOutputColumn::left(cudf_query_engine::ColumnRef::input(0, "id", "left_id"), "left_id"),
            JoinOutputColumn::right(cudf_query_engine::ColumnRef::input(1, "id", "right_id"), "right_id"),
        ]),
        output_schema,
    );
    let plan = builder.finish_with_host_sink(join);
    let context = ExecutionContext::from_shared_runtime(Arc::clone(&runtime));

    match execute_query_plan(&context, &plan) {
        Ok(_) => panic!("native residual validation should reject invalid predicates before source read"),
        Err(error) => error,
    }
}
