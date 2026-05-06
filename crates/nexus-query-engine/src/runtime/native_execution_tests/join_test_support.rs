use super::*;

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
        nexus_query_engine::expr::ColumnRef::input(0, "order_id", "left_order_id"),
        nexus_query_engine::expr::ColumnRef::input(1, "order_id", "right_order_id"),
    )
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
        nexus_query_engine::expr::ColumnRef::input(0, left_name, left_name),
        nexus_query_engine::expr::ColumnRef::input(1, right_name, right_name),
    )
}
