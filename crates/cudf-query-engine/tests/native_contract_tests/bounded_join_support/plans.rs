use super::*;

#[derive(Debug, Clone, Copy)]
pub(super) enum Q22InputSubplanVariant {
    Admitted,
    GroupedAggregate,
    SortInput,
    LimitInput,
    BareLeftAntiInput,
    InnerJoinInput,
    NestedJoinInput,
}

#[path = "plans/remaining_topologies.rs"]
mod remaining_topologies;
#[path = "plans/resident_subplan.rs"]
mod resident_subplan;

pub(super) use remaining_topologies::{
    q02_shaped_aggregate_over_join_subplan_topk_plan, q18_shaped_grouped_aggregate_input_post_join_aggregate_topk_plan,
    q20_shaped_grouped_aggregate_subquery_filter_plan,
};
pub(super) use resident_subplan::{
    KeyedHashJoinResidentSubplanPipelineVariant, keyed_hash_join_resident_subplan_pipeline_plan,
};

pub(super) fn q22_keyless_conditional_join_input_subplan_plan(
    variant: Q22InputSubplanVariant,
) -> cudf_query_engine::QueryPlan {
    let lineitem_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
    ]));
    let orders_schema = Arc::new(Schema::new(vec![Field::new("order_id", DataType::Int32, false)]));
    let left_subplan_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
    ]));
    let grouped_aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("threshold", DataType::Int64, false),
    ]));
    let scalar_aggregate_schema = Arc::new(Schema::new(vec![Field::new("threshold", DataType::Int32, true)]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("qty", DataType::Int32, false),
        Field::new("threshold", DataType::Int32, true),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let lineitem = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/q22-lineitem.parquet",
        Arc::clone(&lineitem_schema),
        vec!["order_id".into(), "qty".into()],
    )));
    let anti_orders = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/q22-anti-orders.parquet",
        Arc::clone(&orders_schema),
        vec!["order_id".into()],
    )));
    let threshold_source = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/q22-threshold.parquet",
        Arc::clone(&orders_schema),
        vec!["order_id".into()],
    )));

    let nested_join_left = if matches!(variant, Q22InputSubplanVariant::NestedJoinInput) {
        let extra_orders = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
            "/tmp/q22-extra-orders.parquet",
            Arc::clone(&orders_schema),
            vec!["order_id".into()],
        )));
        builder.add_join(
            lineitem,
            extra_orders,
            JoinSpec::new(
                JoinType::Inner,
                vec![JoinKey::new(
                    ColumnRef::input(0, "order_id", "left_order_id"),
                    ColumnRef::input(1, "order_id", "right_order_id"),
                )],
            )
            .with_output_projection(vec![
                JoinOutputColumn::left(ColumnRef::input(0, "order_id", "order_id"), "order_id"),
                JoinOutputColumn::left(ColumnRef::input(0, "qty", "qty"), "qty"),
            ])
            .with_build_side(JoinSide::Right),
            Arc::clone(&left_subplan_schema),
        )
    } else {
        lineitem
    };

    let anti_join_type =
        if matches!(variant, Q22InputSubplanVariant::InnerJoinInput) { JoinType::Inner } else { JoinType::LeftAnti };
    let anti_join = builder.add_join(
        nested_join_left,
        anti_orders,
        JoinSpec::new(
            anti_join_type,
            vec![JoinKey::new(
                ColumnRef::input(0, "order_id", "left_order_id"),
                ColumnRef::input(1, "order_id", "right_order_id"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "order_id", "order_id"), "order_id"),
            JoinOutputColumn::left(ColumnRef::input(0, "qty", "qty"), "qty"),
        ])
        .with_build_side(JoinSide::Right),
        Arc::clone(&left_subplan_schema),
    );
    let left_input = if matches!(variant, Q22InputSubplanVariant::BareLeftAntiInput) {
        anti_join
    } else {
        builder.add_projection(
            anti_join,
            projection_outputs(vec![Expr::column("order_id"), Expr::column("qty")], Arc::clone(&left_subplan_schema)),
        )
    };

    let scalar_aggregate = builder.add_aggregate(
        threshold_source,
        AggregateSpec::new(
            Vec::new(),
            vec![AggregateExpr::new(AggregateFunction::Max, Expr::column("order_id"), "threshold")],
        ),
        Arc::clone(&scalar_aggregate_schema),
    );
    let right_input = match variant {
        Q22InputSubplanVariant::GroupedAggregate => builder.add_aggregate(
            threshold_source,
            AggregateSpec::new(
                vec![Expr::column("order_id")],
                vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("order_id"), "threshold")],
            ),
            grouped_aggregate_schema,
        ),
        Q22InputSubplanVariant::SortInput => builder.add_sort(
            scalar_aggregate,
            SortSpec::new(vec![SortExpr::new(Expr::column("threshold"), false, false)]),
            scalar_aggregate_schema,
        ),
        Q22InputSubplanVariant::LimitInput => {
            builder.add_limit(scalar_aggregate, LimitSpec::fetch(1), scalar_aggregate_schema)
        },
        Q22InputSubplanVariant::Admitted
        | Q22InputSubplanVariant::BareLeftAntiInput
        | Q22InputSubplanVariant::InnerJoinInput
        | Q22InputSubplanVariant::NestedJoinInput => scalar_aggregate,
    };

    let root = builder.add_join(
        left_input,
        right_input,
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
        output_schema,
    );
    builder.finish_with_host_sink(root)
}

pub(super) fn q13_shaped_serial_aggregate_spine_plan() -> cudf_query_engine::QueryPlan {
    let customer_schema = Arc::new(Schema::new(vec![Field::new("c_custkey", DataType::Int32, false)]));
    let orders_schema = Arc::new(Schema::new(vec![
        Field::new("o_orderkey", DataType::Int32, false),
        Field::new("o_custkey", DataType::Int32, false),
    ]));
    let join_schema = Arc::new(Schema::new(vec![
        Field::new("c_custkey", DataType::Int32, false),
        Field::new("o_orderkey", DataType::Int32, true),
    ]));
    let first_aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("c_custkey", DataType::Int32, false),
        Field::new("c_count", DataType::Int64, false),
    ]));
    let c_count_schema = Arc::new(Schema::new(vec![Field::new("c_count", DataType::Int64, false)]));
    let final_schema = Arc::new(Schema::new(vec![
        Field::new("c_count", DataType::Int64, false),
        Field::new("custdist", DataType::Int64, false),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let customers = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/q13-customers.parquet",
        customer_schema,
        vec!["c_custkey".into()],
    )));
    let orders = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/q13-orders.parquet",
        orders_schema,
        vec!["o_orderkey".into(), "o_custkey".into()],
    )));
    let join = builder.add_join(
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
        first_aggregate_schema,
    );
    let count_projection = builder.add_projection(
        customer_counts,
        projection_outputs(vec![Expr::column("c_count")], Arc::clone(&c_count_schema)),
    );
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
        Arc::clone(&final_schema),
    );
    builder.finish_with_host_sink(sort)
}

pub(super) fn q16_shaped_serial_aggregate_multi_join_spine_plan() -> cudf_query_engine::QueryPlan {
    let part_schema = Arc::new(Schema::new(vec![
        Field::new("p_partkey", DataType::Int32, false),
        Field::new("p_brand", DataType::Utf8, false),
        Field::new("p_type", DataType::Utf8, false),
        Field::new("p_size", DataType::Int32, false),
    ]));
    let partsupp_schema = Arc::new(Schema::new(vec![
        Field::new("ps_partkey", DataType::Int32, false),
        Field::new("ps_suppkey", DataType::Int32, false),
    ]));
    let supplier_schema = Arc::new(Schema::new(vec![Field::new("s_suppkey", DataType::Int32, false)]));
    let join_schema = Arc::new(Schema::new(vec![
        Field::new("ps_suppkey", DataType::Int32, false),
        Field::new("p_brand", DataType::Utf8, false),
        Field::new("p_type", DataType::Utf8, false),
        Field::new("p_size", DataType::Int32, false),
    ]));
    let distinct_schema = Arc::new(Schema::new(vec![
        Field::new("p_brand", DataType::Utf8, false),
        Field::new("p_type", DataType::Utf8, false),
        Field::new("p_size", DataType::Int32, false),
        Field::new("ps_suppkey", DataType::Int32, false),
    ]));
    let counted_schema = Arc::new(Schema::new(vec![
        Field::new("p_brand", DataType::Utf8, false),
        Field::new("p_type", DataType::Utf8, false),
        Field::new("p_size", DataType::Int32, false),
        Field::new("supplier_cnt", DataType::Int64, false),
    ]));
    let projected_schema = Arc::new(Schema::new(vec![
        Field::new("supplier_cnt", DataType::Int64, false),
        Field::new("p_brand", DataType::Utf8, false),
        Field::new("p_type", DataType::Utf8, false),
        Field::new("p_size", DataType::Int32, false),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let partsupp = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/q16-partsupp.parquet",
        partsupp_schema,
        vec!["ps_partkey".into(), "ps_suppkey".into()],
    )));
    let part = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/q16-part.parquet",
        part_schema,
        vec!["p_partkey".into(), "p_brand".into(), "p_type".into(), "p_size".into()],
    )));
    let supplier = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/q16-supplier.parquet",
        supplier_schema,
        vec!["s_suppkey".into()],
    )));
    let part_join = builder.add_join(
        partsupp,
        part,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "ps_partkey", "ps_partkey"),
                ColumnRef::input(1, "p_partkey", "p_partkey"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "ps_suppkey", "ps_suppkey"), "ps_suppkey"),
            JoinOutputColumn::right(ColumnRef::input(1, "p_brand", "p_brand"), "p_brand"),
            JoinOutputColumn::right(ColumnRef::input(1, "p_type", "p_type"), "p_type"),
            JoinOutputColumn::right(ColumnRef::input(1, "p_size", "p_size"), "p_size"),
        ])
        .with_build_side(JoinSide::Right),
        Arc::clone(&join_schema),
    );
    let supplier_join = builder.add_join(
        part_join,
        supplier,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "ps_suppkey", "ps_suppkey"),
                ColumnRef::input(1, "s_suppkey", "s_suppkey"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "ps_suppkey", "ps_suppkey"), "ps_suppkey"),
            JoinOutputColumn::left(ColumnRef::input(0, "p_brand", "p_brand"), "p_brand"),
            JoinOutputColumn::left(ColumnRef::input(0, "p_type", "p_type"), "p_type"),
            JoinOutputColumn::left(ColumnRef::input(0, "p_size", "p_size"), "p_size"),
        ])
        .with_build_side(JoinSide::Right),
        Arc::clone(&join_schema),
    );
    let distinct_suppliers = builder.add_aggregate(
        supplier_join,
        AggregateSpec::new(
            vec![Expr::column("p_brand"), Expr::column("p_type"), Expr::column("p_size"), Expr::column("ps_suppkey")],
            vec![],
        ),
        distinct_schema,
    );
    let supplier_counts = builder.add_aggregate(
        distinct_suppliers,
        AggregateSpec::new(
            vec![Expr::column("p_brand"), Expr::column("p_type"), Expr::column("p_size")],
            vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("ps_suppkey"), "supplier_cnt")],
        ),
        Arc::clone(&counted_schema),
    );
    let projection = builder.add_projection(
        supplier_counts,
        projection_outputs(
            vec![Expr::column("supplier_cnt"), Expr::column("p_brand"), Expr::column("p_type"), Expr::column("p_size")],
            Arc::clone(&projected_schema),
        ),
    );
    let sort = builder.add_sort(
        projection,
        SortSpec::new(vec![
            SortExpr::new(Expr::column("supplier_cnt"), true, false),
            SortExpr::new(Expr::column("p_brand"), false, false),
            SortExpr::new(Expr::column("p_type"), false, false),
            SortExpr::new(Expr::column("p_size"), false, false),
        ]),
        projected_schema,
    );
    builder.finish_with_host_sink(sort)
}

pub(super) fn single_join_with_aggregate_input_plan(aggregate_side: JoinSide) -> cudf_query_engine::QueryPlan {
    let customer_schema = Arc::new(Schema::new(vec![Field::new("c_custkey", DataType::Int32, false)]));
    let orders_schema = Arc::new(Schema::new(vec![
        Field::new("o_orderkey", DataType::Int32, false),
        Field::new("o_custkey", DataType::Int32, false),
    ]));
    let customer_aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("c_custkey", DataType::Int32, false),
        Field::new("c_count", DataType::Int64, false),
    ]));
    let order_aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("o_custkey", DataType::Int32, false),
        Field::new("o_count", DataType::Int64, false),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let customers = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/q13-customers.parquet",
        customer_schema,
        vec!["c_custkey".into()],
    )));
    let orders = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/q13-orders.parquet",
        orders_schema,
        vec!["o_orderkey".into(), "o_custkey".into()],
    )));
    let customer_counts = builder.add_aggregate(
        customers,
        AggregateSpec::new(
            vec![Expr::column("c_custkey")],
            vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("c_custkey"), "c_count")],
        ),
        Arc::clone(&customer_aggregate_schema),
    );
    let order_counts = builder.add_aggregate(
        orders,
        AggregateSpec::new(
            vec![Expr::column("o_custkey")],
            vec![AggregateExpr::new(AggregateFunction::Count, Expr::column("o_orderkey"), "o_count")],
        ),
        Arc::clone(&order_aggregate_schema),
    );

    let (left, right, left_key, right_key, output_schema, output_projection) = match aggregate_side {
        JoinSide::Left => (
            customer_counts,
            orders,
            ColumnRef::input(0, "c_custkey", "c_custkey"),
            ColumnRef::input(1, "o_custkey", "o_custkey"),
            customer_aggregate_schema,
            vec![
                JoinOutputColumn::left(ColumnRef::input(0, "c_custkey", "c_custkey"), "c_custkey"),
                JoinOutputColumn::left(ColumnRef::input(0, "c_count", "c_count"), "c_count"),
            ],
        ),
        JoinSide::Right => (
            customers,
            order_counts,
            ColumnRef::input(0, "c_custkey", "c_custkey"),
            ColumnRef::input(1, "o_custkey", "o_custkey"),
            order_aggregate_schema,
            vec![
                JoinOutputColumn::right(ColumnRef::input(1, "o_custkey", "o_custkey"), "o_custkey"),
                JoinOutputColumn::right(ColumnRef::input(1, "o_count", "o_count"), "o_count"),
            ],
        ),
    };
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(JoinType::Inner, vec![JoinKey::new(left_key, right_key)])
            .with_output_projection(output_projection)
            .with_build_side(JoinSide::Right),
        output_schema,
    );
    builder.finish_with_host_sink(join)
}
