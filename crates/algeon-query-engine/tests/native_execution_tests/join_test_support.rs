use std::{path::Path, sync::Arc};

use query_engine::{
    admission::admit, error::QueryError, exec::output::QueryExecutionResult, expr::ColumnRef, plan::QueryPlan,
    rapids_backend::rapids_analysis_context,
};

use super::*;

#[path = "join_test_support/mixed_aggregate.rs"]
mod mixed_aggregate;

pub(super) use mixed_aggregate::*;

pub(super) fn assert_bounded_pipeline_admitted(plan: &QueryPlan) {
    let decision = admit(plan, StatisticalAggregateExecutionMode::default(), rapids_analysis_context())
        .expect("bounded admission should decide the plan");
    assert!(decision.admission().is_some(), "plan should be admitted as a compiled pipeline: {decision:?}");
}

pub(super) fn join_left_source(path: &Path, schema: &Arc<Schema>, columns: &[&str]) -> NativeLocalParquetSource {
    fixture_local_parquet_source::from_files(
        vec![LocalParquetFile::new(path).with_file_size_bytes(fs::metadata(path).unwrap().len())],
        Arc::clone(schema),
        columns.iter().map(|column| (*column).to_string()).collect(),
    )
}

pub(super) fn join_right_source(path: &Path, schema: &Arc<Schema>, columns: &[&str]) -> NativeLocalParquetSource {
    fixture_local_parquet_source::from_files(
        vec![LocalParquetFile::new(path).with_file_size_bytes(fs::metadata(path).unwrap().len())],
        Arc::clone(schema),
        columns.iter().map(|column| (*column).to_string()).collect(),
    )
}

pub(super) fn join_order_id_key() -> JoinKey {
    JoinKey::new(ColumnRef::input(0, "order_id", "left_order_id"), ColumnRef::input(1, "order_id", "right_order_id"))
}

pub(super) fn id_join_key() -> JoinKey {
    JoinKey::new(ColumnRef::input(0, "id", "left_id"), ColumnRef::input(1, "id", "right_id"))
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

pub(super) fn bounded_join_execution_context(_runtime: &Arc<cudf::Runtime>) -> BackendQueryContext {
    backend_context_with_grant(1024 * 1024_u64)
}

pub(super) fn execute_bounded_keyless_left_only_conditional_join(
    runtime: &Arc<cudf::Runtime>,
    left_path: &Path,
    right_path: &Path,
    left_schema: &Arc<Schema>,
    right_schema: &Arc<Schema>,
    join_type: JoinType,
    build_side: JoinSide,
) -> QueryExecutionResult {
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
            .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "qty", "qty"), "qty")])
            .with_build_side(build_side),
        Arc::clone(&output_schema),
    );
    let plan = builder.finish_with_host_sink(join);
    assert_bounded_pipeline_admitted(&plan);
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
) -> QueryExecutionResult {
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
            .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "qty", "qty"), "qty")])
            .with_build_side(JoinSide::Right),
        Arc::clone(&output_schema),
    );
    let plan = builder.finish_with_host_sink(join);
    assert_bounded_pipeline_admitted(&plan);
    let context = bounded_join_execution_context(runtime);
    execute_query_plan(&context, &plan).unwrap()
}
#[expect(clippy::too_many_arguments, reason = "test fixture parameters are independent fixture axes")]
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
) -> QueryPlan {
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
                JoinOutputColumn::left(ColumnRef::input(0, "order_id", "order_id"), "order_id"),
                JoinOutputColumn::left(ColumnRef::input(0, "qty", "qty"), "qty"),
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
                JoinOutputColumn::left(ColumnRef::input(0, "order_id", "order_id"), "order_id"),
                JoinOutputColumn::left(ColumnRef::input(0, "qty", "qty"), "qty"),
                JoinOutputColumn::right(ColumnRef::input(1, "threshold", "threshold"), "threshold"),
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
) -> QueryPlan {
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
                JoinOutputColumn::left(ColumnRef::input(0, "ps_partkey", "ps_partkey"), "ps_partkey"),
                JoinOutputColumn::left(ColumnRef::input(0, "qty", "qty"), "qty"),
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
            .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "qty", "qty"), "qty")])
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
                JoinOutputColumn::left(ColumnRef::input(0, "ps_partkey", "ps_partkey"), "ps_partkey"),
                JoinOutputColumn::left(ColumnRef::input(0, "value", "value"), "value"),
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

pub(super) fn supplier_cost_partsupp_source(path: &Path, schema: &Arc<Schema>) -> NativeLocalParquetSource {
    LocalParquetSource::from_files(
        vec![LocalParquetFile::new(path).with_file_size_bytes(fs::metadata(path).unwrap().len())],
        Arc::clone(schema),
        vec!["ps_partkey".to_string(), "ps_suppkey".to_string(), "qty".to_string()],
    )
}

pub(super) fn supplier_cost_supplier_source(path: &Path, schema: &Arc<Schema>) -> NativeLocalParquetSource {
    LocalParquetSource::from_files(
        vec![LocalParquetFile::new(path).with_file_size_bytes(fs::metadata(path).unwrap().len())],
        Arc::clone(schema),
        vec!["s_suppkey".to_string()],
    )
}

pub(super) fn supplier_cost_suppkey_join_key(left_name: &str, right_name: &str) -> JoinKey {
    JoinKey::new(ColumnRef::input(0, left_name, left_name), ColumnRef::input(1, right_name, right_name))
}

pub(super) fn assert_empty_bounded_filtered_join_result(result: &QueryExecutionResult, join_type: &str) {
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
    _runtime: &Arc<cudf::Runtime>,
    left_path: &Path,
    right_path: &Path,
    left_schema: &Arc<Schema>,
    right_schema: &Arc<Schema>,
    join_type: JoinType,
) -> QueryExecutionResult {
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
            .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "qty", "qty"), "qty")])
            .with_build_side(JoinSide::Right),
        Arc::clone(&output_schema),
    );
    let sort = builder.add_sort(
        join,
        SortSpec::new(vec![SortExpr::new(Expr::column("qty"), false, false)]),
        Arc::clone(&output_schema),
    );
    let plan = builder.finish_with_host_sink(sort);
    let context = backend_context();
    execute_query_plan(&context, &plan).unwrap()
}

pub(super) fn execute_bounded_left_only_residual_join(
    runtime: &Arc<cudf::Runtime>,
    left_path: &Path,
    right_path: &Path,
    left_schema: &Arc<Schema>,
    right_schema: &Arc<Schema>,
    join_type: JoinType,
) -> QueryExecutionResult {
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
            .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "qty", "qty"), "qty")])
            .with_build_side(JoinSide::Right),
        Arc::clone(&output_schema),
    );
    let plan = builder.finish_with_host_sink(join);
    assert_bounded_pipeline_admitted(&plan);
    let context = bounded_join_execution_context(runtime);
    execute_query_plan(&context, &plan).unwrap()
}

pub(super) fn residual_validation_error(join_type: JoinType, keys: Vec<JoinKey>, residual: Expr) -> QueryError {
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    residual_validation_error_with_schema(join_type, keys, residual, schema, vec!["id"])
}

pub(super) fn residual_validation_error_with_schema(
    join_type: JoinType,
    keys: Vec<JoinKey>,
    residual: Expr,
    schema: Arc<Schema>,
    read_columns: Vec<&str>,
) -> QueryError {
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("left_id", DataType::Int32, true),
        Field::new("right_id", DataType::Int32, true),
    ]));
    let read_columns = read_columns.into_iter().map(str::to_string).collect::<Vec<_>>();
    let left_source = LocalParquetSource::from_path(
        "/tmp/does-not-need-to-exist-for-residual-validation-left.parquet",
        Arc::clone(&schema),
        read_columns.clone(),
    );
    let right_source = LocalParquetSource::from_path(
        "/tmp/does-not-need-to-exist-for-residual-validation-right.parquet",
        Arc::clone(&schema),
        read_columns,
    );

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(left_source));
    let right = builder.add_source(SourceNode::LocalParquet(right_source));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(join_type, keys).with_residual_predicate(residual).with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "id", "left_id"), "left_id"),
            JoinOutputColumn::right(ColumnRef::input(1, "id", "right_id"), "right_id"),
        ]),
        output_schema,
    );
    let plan = builder.finish_with_host_sink(join);
    let context = backend_context();

    match execute_query_plan(&context, &plan) {
        Ok(_) => panic!("native residual validation should reject invalid predicates before source read"),
        Err(error) => error,
    }
}

pub(super) fn assert_bounded_join_rejection_contract(error: &QueryError) {
    assert_eq!(error.code(), QueryErrorCode::UnsupportedPlanShape);
    assert_eq!(error.operation(), "query_engine.bounded.plan");
    assert_eq!(error.fact_value("reason"), Some("bounded_join_plan_shape_unsupported"));
    for key in [
        "bounded_join_rejection_scope",
        "bounded_join_rejection_owner",
        "bounded_join_rejection_boundary",
        "bounded_join_rejection_category",
        "bounded_join_rejection_reason",
        "bounded_join_rejection_facts",
    ] {
        assert!(error.fact_value(key).is_some(), "bounded join rejection must include {key}");
    }
    for retired_key in [
        "bounded_join_reservation_model",
        "bounded_join_reservation_kinds",
        "bounded_join_reservation_statuses",
        "bounded_join_reservation_requirements",
        "blocked_reservations",
        "missing_reservations",
        "modeled_reservations",
        "deferred_reservations",
        "boundary_policy",
    ] {
        assert_eq!(error.fact_value(retired_key), None, "bounded join rejection must omit {retired_key}");
    }
    assert_eq!(error.fact_value("path"), None);
    assert_eq!(error.fact_value("file_index"), None);
    assert_eq!(error.fact_value("row_group_indexes"), None);
    assert_eq!(error.fact_value("row_group_count"), None);
    assert_eq!(error.fact_value("chunk_index"), None);
    assert!(error.source_error().is_none());
}
