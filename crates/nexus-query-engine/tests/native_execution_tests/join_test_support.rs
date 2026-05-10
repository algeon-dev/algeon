use super::*;

#[path = "join_test_support/mixed_aggregate.rs"]
mod mixed_aggregate;

pub(super) use mixed_aggregate::*;

pub(super) fn join_left_source(path: &Path, schema: &Arc<Schema>, columns: &[&str]) -> NativeLocalParquetSource {
    LocalParquetSource::from_files(
        vec![LocalParquetFile::new(path).with_file_size_bytes(fs::metadata(path).unwrap().len())],
        Arc::clone(schema),
        columns.iter().map(|column| (*column).to_string()).collect(),
    )
}

pub(super) fn join_right_source(path: &Path, schema: &Arc<Schema>, columns: &[&str]) -> NativeLocalParquetSource {
    LocalParquetSource::from_files(
        vec![LocalParquetFile::new(path).with_file_size_bytes(fs::metadata(path).unwrap().len())],
        Arc::clone(schema),
        columns.iter().map(|column| (*column).to_string()).collect(),
    )
}

pub(super) fn join_order_id_key() -> JoinKey {
    JoinKey::new(
        nexus_query_engine::expr::ColumnRef::input(0, "order_id", "left_order_id"),
        nexus_query_engine::expr::ColumnRef::input(1, "order_id", "right_order_id"),
    )
}

pub(super) fn id_join_key() -> JoinKey {
    JoinKey::new(
        nexus_query_engine::expr::ColumnRef::input(0, "id", "left_id"),
        nexus_query_engine::expr::ColumnRef::input(1, "id", "right_id"),
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
) -> nexus_query_engine::exec::output::QueryExecutionResult {
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
                nexus_query_engine::expr::ColumnRef::input(0, "qty", "qty"),
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
) -> nexus_query_engine::exec::output::QueryExecutionResult {
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
                nexus_query_engine::expr::ColumnRef::input(0, "qty", "qty"),
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
) -> nexus_query_engine::plan::QueryPlan {
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
                JoinOutputColumn::left(
                    nexus_query_engine::expr::ColumnRef::input(0, "order_id", "order_id"),
                    "order_id",
                ),
                JoinOutputColumn::left(nexus_query_engine::expr::ColumnRef::input(0, "qty", "qty"), "qty"),
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
                JoinOutputColumn::left(
                    nexus_query_engine::expr::ColumnRef::input(0, "order_id", "order_id"),
                    "order_id",
                ),
                JoinOutputColumn::left(nexus_query_engine::expr::ColumnRef::input(0, "qty", "qty"), "qty"),
                JoinOutputColumn::right(
                    nexus_query_engine::expr::ColumnRef::input(1, "threshold", "threshold"),
                    "threshold",
                ),
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
) -> nexus_query_engine::plan::QueryPlan {
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
                    nexus_query_engine::expr::ColumnRef::input(0, "ps_partkey", "ps_partkey"),
                    "ps_partkey",
                ),
                JoinOutputColumn::left(nexus_query_engine::expr::ColumnRef::input(0, "qty", "qty"), "qty"),
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
                nexus_query_engine::expr::ColumnRef::input(0, "qty", "qty"),
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
                    nexus_query_engine::expr::ColumnRef::input(0, "ps_partkey", "ps_partkey"),
                    "ps_partkey",
                ),
                JoinOutputColumn::left(nexus_query_engine::expr::ColumnRef::input(0, "value", "value"), "value"),
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
    JoinKey::new(
        nexus_query_engine::expr::ColumnRef::input(0, left_name, left_name),
        nexus_query_engine::expr::ColumnRef::input(1, right_name, right_name),
    )
}

pub(super) fn assert_empty_bounded_filtered_join_result(
    result: &nexus_query_engine::exec::output::QueryExecutionResult,
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
    _runtime: &Arc<cudf::Runtime>,
    left_path: &Path,
    right_path: &Path,
    left_schema: &Arc<Schema>,
    right_schema: &Arc<Schema>,
    join_type: JoinType,
) -> nexus_query_engine::exec::output::QueryExecutionResult {
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
                nexus_query_engine::expr::ColumnRef::input(0, "qty", "qty"),
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
) -> nexus_query_engine::exec::output::QueryExecutionResult {
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
                nexus_query_engine::expr::ColumnRef::input(0, "qty", "qty"),
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
) -> nexus_query_engine::error::QueryError {
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    residual_validation_error_with_schema(join_type, keys, residual, schema, vec!["id"])
}

pub(super) fn residual_validation_error_with_schema(
    join_type: JoinType,
    keys: Vec<JoinKey>,
    residual: Expr,
    schema: Arc<Schema>,
    read_columns: Vec<&str>,
) -> nexus_query_engine::error::QueryError {
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
            JoinOutputColumn::left(nexus_query_engine::expr::ColumnRef::input(0, "id", "left_id"), "left_id"),
            JoinOutputColumn::right(nexus_query_engine::expr::ColumnRef::input(1, "id", "right_id"), "right_id"),
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

pub(super) fn assert_bounded_join_rejection_contract(error: &nexus_query_engine::error::QueryError) {
    let contract = BoundedJoinReservationContract;
    let expected_facts = contract.unsupported_shape_facts(ExecutionBoundaryPolicy::default());

    assert_eq!(error.code(), QueryErrorCode::UnsupportedPlanShape);
    assert_eq!(error.operation(), "nexus_query_engine.bounded.plan");
    assert_eq!(error.fact_value("reason"), Some("bounded_join_plan_shape_unsupported"));

    for (key, expected) in &expected_facts {
        if matches!(
            *key,
            "bounded_join_reservation_kinds"
                | "bounded_join_reservation_statuses"
                | "blocked_reservations"
                | "missing_reservations"
                | "modeled_reservations"
                | "bounded_join_lower_level_blockers"
                | "bounded_join_reservation_requirements"
        ) {
            continue;
        }
        assert_eq!(
            error.fact_value(key),
            Some(expected.as_str()),
            "bounded join unsupported-shape fact mismatch for {key}"
        );
    }

    assert_eq!(error.fact_value("bounded_join_reservation_model"), Some(contract.model_status()));
    assert_eq!(error.fact_value("bounded_join_execution_families"), None);
    assert_eq!(error.fact_value("bounded_join_enabled"), Some("true"));
    assert_eq!(
        error.fact_value("boundary_policy"),
        Some(
            "source_read=direct_device_local_parquet,build_materialization=query_scoped_gpu_hash_join,probe_execution=reusable_gpu_hash_join,result_materialization=explicit_sink"
        )
    );
    assert_eq!(error.fact_value("source_read_boundary"), Some("direct_device_local_parquet"));
    assert_eq!(error.fact_value("build_materialization_boundary"), Some("query_scoped_gpu_hash_join"));
    assert_eq!(error.fact_value("probe_execution_boundary"), Some("reusable_gpu_hash_join"));
    assert_eq!(error.fact_value("result_materialization_boundary"), Some("explicit_sink"));
    assert_bounded_join_rejection_reservation_facts_are_consistent(error);
    assert_eq!(error.fact_value("deferred_reservations"), Some(contract.deferred_reservation_kinds_csv().as_str()));
    assert_eq!(error.fact_value("cudf_rs_hash_table_observability"), Some("hash_table_device_allocated_size_bytes"));
    assert_eq!(error.fact_value("cudf_rs_gather_map_observability"), Some("join_indices_device_allocated_size_bytes"));
    assert_eq!(
        error.fact_value("cudf_rs_hash_join_gather_map_reservation_contract"),
        Some("hash_join_gather_map_reservation_size_bytes")
    );
    assert_eq!(error.fact_value("cudf_rs_hash_join_output_size_observability"), Some("hash_join_exact_output_size"));
    assert_eq!(error.fact_value("cudf_rs_hash_join_probe_output_size_hint"), Some("hash_join_probe_with_output_size"));
    assert_eq!(
        error.fact_value("cudf_rs_conditional_join_output_size_observability"),
        Some("conditional_join_exact_output_size")
    );
    assert_eq!(
        error.fact_value("cudf_rs_conditional_join_output_size_hint"),
        Some("conditional_join_indices_with_output_size")
    );
    assert_eq!(
        error.fact_value("cudf_rs_hash_join_pre_build_reservation_contract"),
        Some("hash_join_pre_build_reservation_size_bytes")
    );
    assert_eq!(
        error.fact_value("cudf_rs_hash_join_pre_build_reservation_status"),
        Some("available_exact_flat_hash_join_builds")
    );
    assert_eq!(
        error.fact_value("cudf_rs_filtered_join_hash_table_observability"),
        Some("filtered_join_device_allocated_size_bytes")
    );
    assert_eq!(
        error.fact_value("cudf_rs_filtered_join_output_size_observability"),
        Some("filtered_join_probe_state_output_size")
    );
    assert_eq!(
        error.fact_value("cudf_rs_filtered_join_probe_state_contract"),
        Some("filtered_join_retained_probe_state")
    );
    assert_eq!(
        error.fact_value("cudf_rs_filtered_join_probe_state_observability"),
        Some("filtered_join_probe_device_allocated_size_bytes")
    );
    assert_eq!(
        error.fact_value("cudf_rs_filtered_join_probe_materialization_contract"),
        Some("filtered_join_probe_state_materialize_indices")
    );
    assert_eq!(
        error.fact_value("cudf_rs_dataframe_gather_output_reservation_contract"),
        Some("dataframe_gather_output_reservation_size_bytes")
    );
    assert_eq!(error.fact_value("missing_lower_level_contract"), None);
    assert_eq!(error.fact_value("missing_reservations"), Some(contract.blocking_reservation_kinds_csv().as_str()));
    assert!(error.facts().iter().all(|(key, value)| !key.contains("estimate") && !value.contains("estimate")));
    assert_eq!(error.fact_value("path"), None);
    assert_eq!(error.fact_value("file_index"), None);
    assert_eq!(error.fact_value("row_group_indexes"), None);
    assert_eq!(error.fact_value("row_group_count"), None);
    assert_eq!(error.fact_value("chunk_index"), None);
    assert!(error.source_error().is_none());
}

fn assert_bounded_join_rejection_reservation_facts_are_consistent(error: &nexus_query_engine::error::QueryError) {
    let modeled = error
        .fact_value("modeled_reservations")
        .map(csv_values)
        .expect("bounded join rejection should include modeled reservations");
    let statuses = error
        .fact_value("bounded_join_reservation_statuses")
        .map(csv_pairs)
        .expect("bounded join rejection should include reservation statuses");
    let requirements = error
        .fact_value("bounded_join_reservation_requirements")
        .map(csv_triples)
        .expect("bounded join rejection should include reservation requirements");
    let kinds = error
        .fact_value("bounded_join_reservation_kinds")
        .map(csv_values)
        .expect("bounded join rejection should include reservation kinds");

    assert_eq!(statuses.iter().map(|(kind, _)| *kind).collect::<Vec<_>>(), kinds);
    assert_eq!(requirements.iter().map(|(kind, _, _)| *kind).collect::<Vec<_>>(), kinds);
    assert_eq!(
        modeled,
        statuses.iter().filter_map(|(kind, status)| (*status == "modeled").then_some(*kind)).collect::<Vec<_>>()
    );
    assert_eq!(error.fact_value("blocked_reservations"), Some(""));
    assert_eq!(error.fact_value("missing_reservations"), Some(""));
    assert_eq!(error.fact_value("bounded_join_lower_level_blockers"), Some(""));
}

fn csv_values(value: &str) -> Vec<&str> {
    value.split(',').filter(|value| !value.is_empty()).collect()
}

fn csv_pairs(value: &str) -> Vec<(&str, &str)> {
    value.split(',').filter_map(|row| row.split_once(':')).collect()
}

fn csv_triples(value: &str) -> Vec<(&str, &str, &str)> {
    value
        .split(',')
        .filter_map(|row| {
            let (left, detail) = row.rsplit_once(':')?;
            let (kind, contract) = left.split_once(':')?;
            Some((kind, contract, detail))
        })
        .collect()
}
