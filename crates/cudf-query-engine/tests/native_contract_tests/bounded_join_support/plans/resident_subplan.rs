use super::*;

#[derive(Debug, Clone, Copy)]
pub(in crate::bounded_join_support) enum KeyedHashJoinResidentSubplanPipelineVariant {
    BlockedFamily,
    ResidualRootJoin,
    SortBelowRoot,
    ScalarAggregateNotMax,
    GroupedAggregateNotSum,
}

pub(in crate::bounded_join_support) fn keyed_hash_join_resident_subplan_pipeline_plan(
    variant: KeyedHashJoinResidentSubplanPipelineVariant,
) -> cudf_query_engine::QueryPlan {
    let supplier_schema = Arc::new(Schema::new(vec![
        Field::new("s_suppkey", DataType::Int32, false),
        Field::new("s_name", DataType::Utf8, false),
    ]));
    let lineitem_schema = Arc::new(Schema::new(vec![
        Field::new("l_suppkey", DataType::Int32, false),
        Field::new("l_shipdate", DataType::Int32, false),
        Field::new("l_extendedprice", DataType::Int64, false),
    ]));
    let revenue_projection_schema = Arc::new(Schema::new(vec![
        Field::new("supplier_no", DataType::Int32, false),
        Field::new("revenue", DataType::Int64, false),
    ]));
    let revenue_schema = Arc::new(Schema::new(vec![
        Field::new("supplier_no", DataType::Int32, false),
        Field::new("total_revenue", DataType::Int64, true),
    ]));
    let scalar_revenue_input_schema = Arc::new(Schema::new(vec![Field::new("total_revenue", DataType::Int64, true)]));
    let supplier_revenue_schema = Arc::new(Schema::new(vec![
        Field::new("s_suppkey", DataType::Int32, false),
        Field::new("s_name", DataType::Utf8, false),
        Field::new("total_revenue", DataType::Int64, true),
    ]));
    let max_revenue_schema = Arc::new(Schema::new(vec![Field::new("max_revenue", DataType::Int64, true)]));
    let grouped_revenue_function =
        if matches!(variant, KeyedHashJoinResidentSubplanPipelineVariant::GroupedAggregateNotSum) {
            AggregateFunction::Count
        } else {
            AggregateFunction::Sum
        };
    let scalar_revenue_function =
        if matches!(variant, KeyedHashJoinResidentSubplanPipelineVariant::ScalarAggregateNotMax) {
            AggregateFunction::Sum
        } else {
            AggregateFunction::Max
        };

    let mut builder = QueryPlanBuilder::new();
    let supplier = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/resident-subplan-supplier.parquet",
        Arc::clone(&supplier_schema),
        vec!["s_suppkey".into(), "s_name".into()],
    )));
    let revenue = build_resident_subplan_revenue_branch(
        &mut builder,
        "left",
        matches!(variant, KeyedHashJoinResidentSubplanPipelineVariant::SortBelowRoot),
        Arc::clone(&lineitem_schema),
        Arc::clone(&revenue_projection_schema),
        Arc::clone(&revenue_schema),
        grouped_revenue_function,
    );
    let supplier_revenue = builder.add_join(
        supplier,
        revenue,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "s_suppkey", "s_suppkey"),
                ColumnRef::input(1, "supplier_no", "supplier_no"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "s_suppkey", "s_suppkey"), "s_suppkey"),
            JoinOutputColumn::left(ColumnRef::input(0, "s_name", "s_name"), "s_name"),
            JoinOutputColumn::right(ColumnRef::input(1, "total_revenue", "total_revenue"), "total_revenue"),
        ])
        .with_build_side(JoinSide::Right),
        Arc::clone(&supplier_revenue_schema),
    );
    let supplier_revenue = builder.add_projection(
        supplier_revenue,
        projection_outputs(
            vec![Expr::column("s_suppkey"), Expr::column("s_name"), Expr::column("total_revenue")],
            Arc::clone(&supplier_revenue_schema),
        ),
    );

    let revenue_for_max = build_resident_subplan_revenue_aggregate(
        &mut builder,
        "right",
        false,
        lineitem_schema,
        revenue_projection_schema,
        Arc::clone(&revenue_schema),
        grouped_revenue_function,
    );
    let revenue_for_max = builder.add_projection(
        revenue_for_max,
        projection_outputs(vec![Expr::column("total_revenue")], Arc::clone(&scalar_revenue_input_schema)),
    );
    let max_revenue = builder.add_aggregate(
        revenue_for_max,
        AggregateSpec::new(
            Vec::new(),
            vec![AggregateExpr::new(scalar_revenue_function, Expr::column("total_revenue"), "max_revenue")],
        ),
        Arc::clone(&max_revenue_schema),
    );

    let mut root_spec = JoinSpec::new(
        JoinType::Inner,
        vec![JoinKey::new(
            ColumnRef::input(0, "total_revenue", "total_revenue"),
            ColumnRef::input(1, "max_revenue", "max_revenue"),
        )],
    )
    .with_output_projection(vec![
        JoinOutputColumn::left(ColumnRef::input(0, "s_suppkey", "s_suppkey"), "s_suppkey"),
        JoinOutputColumn::left(ColumnRef::input(0, "s_name", "s_name"), "s_name"),
        JoinOutputColumn::left(ColumnRef::input(0, "total_revenue", "total_revenue"), "total_revenue"),
    ])
    .with_build_side(JoinSide::Right);
    if matches!(variant, KeyedHashJoinResidentSubplanPipelineVariant::ResidualRootJoin) {
        root_spec = root_spec.with_residual_predicate(Expr::binary(
            Expr::input_column(0, "s_suppkey", "s_suppkey"),
            BinaryOp::Gt,
            Expr::literal(LiteralValue::Int32(0)),
        ));
    }
    let root = builder.add_join(supplier_revenue, max_revenue, root_spec, Arc::clone(&supplier_revenue_schema));
    let output = builder.add_sort(
        root,
        SortSpec::new(vec![SortExpr::new(Expr::column("s_suppkey"), false, false)]),
        supplier_revenue_schema,
    );
    builder.finish_with_host_sink(output)
}

fn build_resident_subplan_revenue_branch(
    builder: &mut QueryPlanBuilder,
    label: &str,
    sort_below_root: bool,
    lineitem_schema: Arc<Schema>,
    revenue_projection_schema: Arc<Schema>,
    revenue_schema: Arc<Schema>,
    aggregate_function: AggregateFunction,
) -> cudf_query_engine::NodeId {
    let aggregate = build_resident_subplan_revenue_aggregate(
        builder,
        label,
        sort_below_root,
        lineitem_schema,
        revenue_projection_schema,
        Arc::clone(&revenue_schema),
        aggregate_function,
    );
    builder.add_projection(
        aggregate,
        projection_outputs(vec![Expr::column("supplier_no"), Expr::column("total_revenue")], revenue_schema),
    )
}

fn build_resident_subplan_revenue_aggregate(
    builder: &mut QueryPlanBuilder,
    label: &str,
    sort_below_root: bool,
    lineitem_schema: Arc<Schema>,
    revenue_projection_schema: Arc<Schema>,
    revenue_schema: Arc<Schema>,
    aggregate_function: AggregateFunction,
) -> cudf_query_engine::NodeId {
    let lineitem = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        format!("/tmp/resident-subplan-{label}-lineitem.parquet"),
        Arc::clone(&lineitem_schema),
        vec!["l_suppkey".into(), "l_shipdate".into(), "l_extendedprice".into()],
    )));
    let filtered = builder.add_filter(
        lineitem,
        Expr::binary(Expr::column("l_shipdate"), BinaryOp::GtEq, Expr::literal(LiteralValue::Int32(0))),
        Arc::clone(&lineitem_schema),
    );
    let projected = builder.add_projection(
        filtered,
        projection_outputs(
            vec![Expr::column("l_suppkey").alias("supplier_no"), Expr::column("l_extendedprice").alias("revenue")],
            Arc::clone(&revenue_projection_schema),
        ),
    );
    let aggregate_input = if sort_below_root {
        builder.add_sort(
            projected,
            SortSpec::new(vec![SortExpr::new(Expr::column("supplier_no"), false, false)]),
            revenue_projection_schema,
        )
    } else {
        projected
    };
    builder.add_aggregate(
        aggregate_input,
        AggregateSpec::new(
            vec![Expr::column("supplier_no")],
            vec![AggregateExpr::new(aggregate_function, Expr::column("revenue"), "total_revenue")],
        ),
        revenue_schema,
    )
}
