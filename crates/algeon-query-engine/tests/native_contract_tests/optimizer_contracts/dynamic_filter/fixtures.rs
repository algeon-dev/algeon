fn build_subtree_source_reuse_dynamic_filter_cycle_plan() -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let shared_probe =
        builder.add_source(estimated_local_source("shared_probe", "shared_key", 8 * 1024 * 1024, 10_000));
    let build_seed = builder.add_source(estimated_local_source("build_seed", "build_key", 8 * 1024 * 1024, 10_000));
    let build_subtree_schema = schema(&["build_key"]);
    let build_subtree = builder.add_join(
        build_seed,
        shared_probe,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "build_key", "build_key"),
                ColumnRef::input(1, "shared_key", "shared_key"),
            )],
        )
        .with_output_projection(vec![JoinOutputColumn::left(
            ColumnRef::input(0, "build_key", "build_key"),
            "build_key",
        )])
        .with_build_side(JoinSide::Right),
        Arc::clone(&build_subtree_schema),
    ).unwrap();
    let root = builder.add_join(
        shared_probe,
        build_subtree,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "shared_key", "shared_key"),
                ColumnRef::input(1, "build_key", "build_key"),
            )],
        )
        .with_output_projection(vec![JoinOutputColumn::left(
            ColumnRef::input(0, "shared_key", "shared_key"),
            "shared_key",
        )])
        .with_build_side(JoinSide::Right),
        schema(&["shared_key"]),
    ).unwrap();
    builder.finish_gpu_resident(root).unwrap()
}

fn repeated_consumer_file_set_dynamic_filter_plan() -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let output_schema = schema(&["left_key"]);
    let fact = builder.add_source(estimated_local_source_with_stable_id(
        "fact_probe",
        "shared_fact",
        &["left_key"],
        512 * 1024 * 1024,
        10_000_000,
    ));
    let repeated_fact = builder.add_source(estimated_local_source_with_stable_id(
        "fact_repeated",
        "shared_fact",
        &["left_key"],
        512 * 1024 * 1024,
        10_000_000,
    ));
    let dim = builder.add_source(estimated_local_source("dim", "right_key", 16 * 1024 * 1024, 100_000));
    let join = builder.add_join(
        fact,
        dim,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "left_key", "left_key"),
                ColumnRef::input(1, "right_key", "right_key"),
            )],
        )
        .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "left_key", "left_key"), "left_key")])
        .with_build_side(JoinSide::Right),
        Arc::clone(&output_schema),
    ).unwrap();
    let union = builder.add_union(vec![join, repeated_fact], output_schema).unwrap();
    builder.finish_gpu_resident(union).unwrap()
}

fn shared_prefix_branch_local_dynamic_filter_plan() -> (QueryPlan, NodeId, NodeId) {
    let mut builder = QueryPlanBuilder::new();
    let fact = builder.add_source(estimated_local_source_with_fields(
        "store_sales",
        &["sold_date_sk", "item_sk"],
        512 * 1024 * 1024,
        10_000_000,
    ));
    let shared_prefix = builder.add_projection(
        fact,
        vec![
            ProjectionOutput::new(Expr::column("sold_date_sk"), Field::new("sold_date_sk", DataType::Int32, false)),
            ProjectionOutput::new(Expr::column("item_sk"), Field::new("item_sk", DataType::Int32, false)),
        ],
    ).unwrap();
    let first_date = builder.add_source(estimated_local_source("date_1999", "date_sk", 16 * 1024 * 1024, 366));
    let second_date = builder.add_source(estimated_local_source("date_2000", "date_sk", 16 * 1024 * 1024, 366));
    let branch_schema = schema(&["item_sk"]);
    let first_branch_join = builder.add_join(
        shared_prefix,
        first_date,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "sold_date_sk", "sold_date_sk"),
                ColumnRef::input(1, "date_sk", "date_sk"),
            )],
        )
        .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "item_sk", "item_sk"), "item_sk")])
        .with_build_side(JoinSide::Right),
        Arc::clone(&branch_schema),
    ).unwrap();
    let second_branch_join = builder.add_join(
        shared_prefix,
        second_date,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "sold_date_sk", "sold_date_sk"),
                ColumnRef::input(1, "date_sk", "date_sk"),
            )],
        )
        .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "item_sk", "item_sk"), "item_sk")])
        .with_build_side(JoinSide::Right),
        Arc::clone(&branch_schema),
    ).unwrap();
    let final_join = builder.add_join(
        first_branch_join,
        second_branch_join,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "item_sk", "item_sk"), ColumnRef::input(1, "item_sk", "item_sk"))],
        )
        .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "item_sk", "item_sk"), "item_sk")])
        .with_build_side(JoinSide::Right),
        branch_schema,
    ).unwrap();
    (builder.finish_gpu_resident(final_join).unwrap(), first_branch_join, second_branch_join)
}

fn shared_filtered_prefix_branch_local_dynamic_filter_plan() -> (QueryPlan, NodeId, NodeId, NodeId) {
    let mut builder = QueryPlanBuilder::new();
    let fact = builder.add_source(estimated_local_source_with_fields(
        "store_sales",
        &["sold_date_sk", "item_sk"],
        512 * 1024 * 1024,
        10_000_000,
    ));
    let shared_filter = builder.add_filter(
        fact,
        Expr::binary(Expr::column("item_sk"), BinaryOp::GtEq, Expr::literal(LiteralValue::Int32(0)))).unwrap();
    let shared_prefix = builder.add_projection(
        shared_filter,
        vec![
            ProjectionOutput::new(Expr::column("sold_date_sk"), Field::new("sold_date_sk", DataType::Int32, false)),
            ProjectionOutput::new(Expr::column("item_sk"), Field::new("item_sk", DataType::Int32, false)),
        ],
    ).unwrap();
    let first_date = builder.add_source(estimated_local_source("date_1999", "date_sk", 16 * 1024 * 1024, 366));
    let second_date = builder.add_source(estimated_local_source("date_2000", "date_sk", 16 * 1024 * 1024, 366));
    let branch_schema = schema(&["item_sk"]);
    let first_branch_join = builder.add_join(
        shared_prefix,
        first_date,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "sold_date_sk", "sold_date_sk"),
                ColumnRef::input(1, "date_sk", "date_sk"),
            )],
        )
        .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "item_sk", "item_sk"), "item_sk")])
        .with_build_side(JoinSide::Right),
        Arc::clone(&branch_schema),
    ).unwrap();
    let second_branch_join = builder.add_join(
        shared_prefix,
        second_date,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "sold_date_sk", "sold_date_sk"),
                ColumnRef::input(1, "date_sk", "date_sk"),
            )],
        )
        .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "item_sk", "item_sk"), "item_sk")])
        .with_build_side(JoinSide::Right),
        Arc::clone(&branch_schema),
    ).unwrap();
    let final_join = builder.add_join(
        first_branch_join,
        second_branch_join,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "item_sk", "item_sk"), ColumnRef::input(1, "item_sk", "item_sk"))],
        )
        .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "item_sk", "item_sk"), "item_sk")])
        .with_build_side(JoinSide::Right),
        branch_schema,
    ).unwrap();
    (builder.finish_gpu_resident(final_join).unwrap(), shared_filter, first_branch_join, second_branch_join)
}

struct SharedPartsuppDynamicFilterPlan {
    plan: QueryPlan,
    partsupp: NodeId,
    part_partsupp_join: NodeId,
}

#[derive(Clone)]
struct SharedPartsuppDynamicFilterPlanOptions {
    extra_unconstrained_partsupp_consumer: bool,
    final_join_type: JoinType,
    final_join_null_equality: JoinNullEquality,
    derived_aggregate_group_key: bool,
    filter_part_build_keyset: bool,
    final_join_uses_unfiltered_part_keyset: bool,
    limited_constrained_partsupp_consumer: bool,
}

impl Default for SharedPartsuppDynamicFilterPlanOptions {
    fn default() -> Self {
        Self {
            extra_unconstrained_partsupp_consumer: false,
            final_join_type: JoinType::Inner,
            final_join_null_equality: JoinNullEquality::NullsDoNotMatch,
            derived_aggregate_group_key: false,
            filter_part_build_keyset: false,
            final_join_uses_unfiltered_part_keyset: false,
            limited_constrained_partsupp_consumer: false,
        }
    }
}

fn q02_like_shared_partsupp_dynamic_filter_plan(
    options: SharedPartsuppDynamicFilterPlanOptions,
) -> SharedPartsuppDynamicFilterPlan {
    let mut builder = QueryPlanBuilder::new();
    let result_schema = schema(&["p_partkey", "ps_supplycost"]);
    let part = builder.add_source(estimated_local_source("part", "p_partkey", 16 * 1024 * 1024, 100_000));
    let part_build_input = if options.filter_part_build_keyset {
        builder.add_filter(
            part,
            Expr::binary(Expr::column("p_partkey"), BinaryOp::GtEq, Expr::literal(LiteralValue::Int32(0)))).unwrap()
    } else {
        part
    };
    let partsupp = builder.add_source(estimated_local_source_with_fields(
        "partsupp",
        &["ps_partkey", "ps_suppkey", "ps_supplycost"],
        512 * 1024 * 1024,
        80_000_000,
    ));
    let part_partsupp_join = builder.add_join(
        part_build_input,
        partsupp,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "p_partkey", "p_partkey"),
                ColumnRef::input(1, "ps_partkey", "ps_partkey"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "p_partkey", "p_partkey"), "p_partkey"),
            JoinOutputColumn::right(ColumnRef::input(1, "ps_supplycost", "ps_supplycost"), "ps_supplycost"),
        ])
        .with_build_side(JoinSide::Left),
        Arc::clone(&result_schema),
    ).unwrap();
    let (aggregate_key_expr, aggregate_key_name) = if options.derived_aggregate_group_key {
        (
            Expr::binary(Expr::column("ps_partkey"), BinaryOp::Add, Expr::literal(LiteralValue::Int32(0)))
                .alias("ps_partkey_plus_zero"),
            "ps_partkey_plus_zero",
        )
    } else {
        (Expr::column("ps_partkey"), "ps_partkey")
    };
    let supplier = builder.add_source(estimated_local_source("supplier", "s_suppkey", 8 * 1024 * 1024, 10_000));
    let supplier_partsupp = builder.add_join(
        supplier,
        partsupp,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "s_suppkey", "s_suppkey"),
                ColumnRef::input(1, "ps_suppkey", "ps_suppkey"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::right(ColumnRef::input(1, "ps_partkey", "ps_partkey"), "ps_partkey"),
            JoinOutputColumn::right(ColumnRef::input(1, "ps_supplycost", "ps_supplycost"), "ps_supplycost"),
        ])
        .with_build_side(JoinSide::Left),
        Arc::clone(&result_schema),
    ).unwrap();
    let partsupp_min = builder.add_aggregate(
        supplier_partsupp,
        AggregateSpec::new(
            vec![aggregate_key_expr],
            vec![AggregateExpr::new(AggregateFunction::Min, Expr::column("ps_supplycost"), "min_supplycost")],
        ),
        schema(&[aggregate_key_name, "min_supplycost"]),
    ).unwrap();
    let final_join_left = if options.final_join_uses_unfiltered_part_keyset {
        builder.add_projection(
            part,
            vec![
                ProjectionOutput::new(Expr::column("p_partkey"), Field::new("p_partkey", DataType::Int32, false)),
                ProjectionOutput::new(
                    Expr::literal(LiteralValue::Int32(0)),
                    Field::new("ps_supplycost", DataType::Int32, false),
                ),
            ],
        ).unwrap()
    } else {
        part_partsupp_join
    };
    let final_join = builder.add_join(
        final_join_left,
        partsupp_min,
        JoinSpec::new(
            options.final_join_type,
            vec![
                JoinKey::new(
                    ColumnRef::input(0, "p_partkey", "p_partkey"),
                    ColumnRef::input(1, aggregate_key_name, aggregate_key_name),
                ),
                JoinKey::new(
                    ColumnRef::input(0, "ps_supplycost", "ps_supplycost"),
                    ColumnRef::input(1, "min_supplycost", "min_supplycost"),
                ),
            ],
        )
        .with_null_equality(options.final_join_null_equality)
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "p_partkey", "p_partkey"), "p_partkey"),
            JoinOutputColumn::left(ColumnRef::input(0, "ps_supplycost", "ps_supplycost"), "ps_supplycost"),
        ])
        .with_build_side(JoinSide::Right),
        Arc::clone(&result_schema),
    ).unwrap();
    let mut root_inputs = vec![final_join];
    if options.final_join_uses_unfiltered_part_keyset {
        root_inputs.push(part_partsupp_join);
    }
    if options.extra_unconstrained_partsupp_consumer {
        let extra_consumer = builder.add_projection(
            partsupp,
            vec![
                ProjectionOutput::new(Expr::column("ps_partkey"), Field::new("p_partkey", DataType::Int32, false)),
                ProjectionOutput::new(
                    Expr::column("ps_supplycost"),
                    Field::new("ps_supplycost", DataType::Int32, false),
                ),
            ],
        ).unwrap();
        root_inputs.push(extra_consumer);
    }
    if options.limited_constrained_partsupp_consumer {
        // Extra shared-partsupp consumer whose key is constrained by the SAME `part` keyset, but
        // only AFTER an intervening `Limit`. Filtering the shared source in place would change the
        // rows the `Limit` selects, so this branch must NOT be treated as keyset-constrained.
        let limited_partsupp = builder.add_limit(partsupp, LimitSpec::fetch(10)).unwrap();
        let limited_constrained_join = builder.add_join(
            limited_partsupp,
            part,
            JoinSpec::new(
                JoinType::Inner,
                vec![JoinKey::new(
                    ColumnRef::input(0, "ps_partkey", "ps_partkey"),
                    ColumnRef::input(1, "p_partkey", "p_partkey"),
                )],
            )
            .with_output_projection(vec![
                JoinOutputColumn::left(ColumnRef::input(0, "ps_partkey", "ps_partkey"), "p_partkey"),
                JoinOutputColumn::left(ColumnRef::input(0, "ps_supplycost", "ps_supplycost"), "ps_supplycost"),
            ])
            .with_build_side(JoinSide::Right),
            Arc::clone(&result_schema),
        ).unwrap();
        root_inputs.push(limited_constrained_join);
    }
    let root =
        if root_inputs.len() == 1 { final_join } else { builder.add_union(root_inputs, Arc::clone(&result_schema)).unwrap() };

    SharedPartsuppDynamicFilterPlan { plan: builder.finish_gpu_resident(root).unwrap(), partsupp, part_partsupp_join }
}

fn aggregate_keyset_source(builder: &mut QueryPlanBuilder, name: &str, key_name: &str) -> NodeId {
    let source = builder.add_source(gpu_source(name, &[key_name]));
    builder.add_aggregate(source, AggregateSpec::new(vec![Expr::column(key_name)], Vec::new()), schema(&[key_name])).unwrap()
}

fn expr_contains_int32_equality(expr: &Expr, column: &str, value: i32) -> bool {
    match expr {
        Expr::Binary { left, op: BinaryOp::Eq, right, .. } => {
            (expr_is_column(left, column) && expr_is_int32_literal(right, value))
                || (expr_is_column(right, column) && expr_is_int32_literal(left, value))
                || expr_contains_int32_equality(left, column, value)
                || expr_contains_int32_equality(right, column, value)
        },
        Expr::Binary { left, right, .. } => {
            expr_contains_int32_equality(left, column, value) || expr_contains_int32_equality(right, column, value)
        },
        Expr::Not(inner) | Expr::IsNull(inner) | Expr::IsNotNull(inner) => {
            expr_contains_int32_equality(inner, column, value)
        },
        Expr::Cast { expr, .. } | Expr::TryCast { expr, .. } => expr_contains_int32_equality(expr, column, value),
        Expr::Alias { expr, .. } => expr_contains_int32_equality(expr, column, value),
        _ => false,
    }
}

fn expr_is_column(expr: &Expr, column: &str) -> bool {
    matches!(expr, Expr::Column(expr_column) if expr_column.name() == column)
}

fn expr_is_int32_literal(expr: &Expr, value: i32) -> bool {
    matches!(expr, Expr::Literal(LiteralValue::Int32(candidate)) if *candidate == value)
}

fn max_direct_join_probe_consumer_count(plan: &QueryPlan) -> usize {
    let mut counts = BTreeMap::<NodeId, usize>::new();
    for node in plan.nodes() {
        let PlanNode::Join(join) = &node.kind else {
            continue;
        };
        let probe_input = match join.spec.build_side {
            JoinSide::Left => join.inputs[1],
            JoinSide::Right => join.inputs[0],
        };
        *counts.entry(probe_input).or_default() += 1;
    }
    counts.into_values().max().unwrap_or(0)
}
