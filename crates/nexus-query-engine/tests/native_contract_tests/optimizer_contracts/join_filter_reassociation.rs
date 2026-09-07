#[test]
fn join_filter_reassociation_preserves_output_advisory_for_residual_build_selection() {
    for join_type in [JoinType::LeftSemi, JoinType::LeftAnti] {
        for (parent_rows, right_bytes, right_rows, expected_side) in [
            (Some(2_000), 5 * 1024 * 1024 * 1024, 100_000_000, JoinSide::Left),
            (Some(2_000), 32 * 1024, 1_000, JoinSide::Right),
            (None, 5 * 1024 * 1024 * 1024, 100_000_000, JoinSide::Right),
        ] {
            let parent_advisory = parent_rows.map(|rows| AdvisoryOutputStatistics {
                bytes: AdvisoryStatistic::Inexact(64_000),
                ..advisory_rows(rows)
            });
            let (mut builder, left) = inner_join_filter_over_projected_inner_join_builder(
                parent_advisory,
                Some(advisory_rows(500_000_000)),
            );
            let right = builder.add_source(estimated_local_source_with_fields(
                "probe",
                &["probe_key", "probe_value"],
                right_bytes,
                right_rows,
            ));
            let root = builder.add_join(
                left,
                right,
                JoinSpec::new(
                    join_type,
                    vec![JoinKey::new(
                        ColumnRef::input(0, "customer", "customer"),
                        ColumnRef::input(1, "probe_key", "probe_key"),
                    )],
                )
                .with_residual_predicate(Expr::binary(
                    Expr::input_column(0, "net_amount", "net_amount"),
                    BinaryOp::NotEq,
                    Expr::input_column(1, "probe_value", "probe_value"),
                ))
                .with_output_projection(vec![JoinOutputColumn::left(
                    ColumnRef::input(0, "customer", "customer"),
                    "customer",
                )])
                .with_build_side(JoinSide::Right),
                schema(&["customer"]),
            );
            let result = optimize_query_plan(builder.finish_gpu_resident(root), &NativeOptimizerOptions::default(), rapids_analysis_context());
            assert!(result.diagnostics.join_filter_reassociation.applied_count > 0);
            let candidate = result
                .diagnostics
                .join_build_side_selection
                .candidates
                .iter()
                .find(|candidate| candidate.right_estimated_rows == Some(right_rows))
                .expect("residual join should retain its known probe row count");
            assert_eq!(candidate.left_estimated_rows, None);
            assert_eq!(candidate.left_estimated_bytes, None);
            assert_eq!(candidate.left_advisory_rows, parent_rows);
            assert_eq!(candidate.left_advisory_bytes, None, "reassociation changes the intermediate output columns");
            assert_eq!(root_join_build_side(&result.plan), expected_side);
        }
    }
}

#[test]
fn join_filter_reassociation_pushes_dimension_filter_below_fact_join_without_losing_multiplicity() {
    let plan = inner_join_filter_over_projected_inner_join_plan();

    let result = optimize_query_plan(plan, &NativeOptimizerOptions::default(), rapids_analysis_context());

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    assert_eq!(result.diagnostics.join_filter_reassociation.applied_count, 1);
    let candidate = &result.diagnostics.join_filter_reassociation.candidates[0];
    assert!(candidate.applied);
    assert_eq!(candidate.rejected_reason, None);

    let PlanNode::Projection(parent_projection) = &result.plan.root_node().kind else {
        panic!("rewritten parent join should become a projection preserving the original output schema");
    };
    let PlanNode::Projection(case_projection) =
        &result.plan.node(parent_projection.input).expect("case projection").kind
    else {
        panic!("parent projection should consume rebuilt case projection");
    };
    let PlanNode::Join(fact_join) = &result.plan.node(case_projection.input).expect("rebuilt fact join").kind else {
        panic!("case projection should consume rebuilt fact join");
    };
    assert_eq!(fact_join.spec.join_type, JoinType::Inner);
    let PlanNode::Join(filter_join) = &result.plan.node(fact_join.inputs[1]).expect("filter join").kind else {
        panic!("fact join right input should be prefiltered by dimension");
    };
    assert_eq!(filter_join.spec.join_type, JoinType::Inner);
    assert_source_input(&result.plan, filter_join.inputs[0], "returns");
    assert_source_input(&result.plan, filter_join.inputs[1], "reason");
}

#[test]
fn join_filter_reassociation_repeats_until_filter_reaches_base_fact_input() {
    let plan = nested_inner_join_filter_plan();

    let result = optimize_query_plan(plan, &NativeOptimizerOptions::default(), rapids_analysis_context());

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    assert_eq!(result.diagnostics.join_filter_reassociation.applied_count, 2);
    assert!(
        result
            .diagnostics
            .join_filter_reassociation
            .candidates
            .iter()
            .filter(|candidate| candidate.applied)
            .all(|candidate| candidate.rejected_reason.is_none())
    );
    assert!(
        result.plan.nodes().iter().any(|node| {
            let PlanNode::Join(join) = &node.kind else { return false };
            join.spec.join_type == JoinType::Inner
                && join.spec.keys.len() == 1
                && source_input_name(&result.plan, join.inputs[0]).as_deref() == Some("sales")
                && source_input_name(&result.plan, join.inputs[1]).as_deref() == Some("date")
        }),
        "fixed-point reassociation should push the selective date join down to the sales source"
    );
}

#[test]
fn join_filter_reassociation_uses_parent_projection_to_filter_join_child_before_fact_join() {
    let plan = projected_parent_join_filter_over_fact_join_plan();

    let result = optimize_query_plan(plan, &NativeOptimizerOptions::default(), rapids_analysis_context());

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    assert!(
        result.diagnostics.join_filter_reassociation.applied_count >= 1,
        "projected parent join should be reassociated before the large fact join:\n{}",
        result.plan
    );
    assert!(
        result.plan.nodes().iter().any(|node| {
            let PlanNode::Join(join) = &node.kind else { return false };
            join.spec.keys.len() == 1
                && join.spec.keys[0].left.name() == "s_nationkey"
                && join.spec.keys[0].right.name() == "n_nationkey"
                && subtree_contains_gpu_source(&result.plan, join.inputs[0], "supplier")
                && subtree_contains_gpu_source(&result.plan, join.inputs[1], "nation")
        }),
        "filtered dimension join should be pushed to supplier before supplier joins partsupp:\n{}",
        result.plan
    );
    assert!(
        result.plan.nodes().iter().any(|node| {
            let PlanNode::Join(join) = &node.kind else { return false };
            join.spec.keys.len() == 1
                && join.spec.keys[0].left.name() == "s_suppkey"
                && join.spec.keys[0].right.name() == "ps_suppkey"
                && subtree_contains_gpu_source(&result.plan, join.inputs[0], "supplier")
                && subtree_contains_gpu_source(&result.plan, join.inputs[0], "nation")
                && subtree_contains_gpu_source(&result.plan, join.inputs[1], "partsupp")
        }),
        "partsupp should join after supplier has been filtered by nation/region:\n{}",
        result.plan
    );
}

#[test]
fn join_filter_reassociation_rejects_shared_projected_parent_to_preserve_projection_fanout() {
    let plan = shared_projected_parent_join_filter_plan();

    let result = optimize_query_plan(plan, &NativeOptimizerOptions::default(), rapids_analysis_context());

    assert_eq!(result.diagnostics.join_filter_reassociation.applied_count, 0);
    assert!(
        result.diagnostics.join_filter_reassociation.candidates.iter().any(|candidate| {
            !candidate.applied
                && candidate.rejected_reason == Some(JoinFilterReassociationRejectedReason::SharedProjectedParent)
        }),
        "shared projected parent should be rejected before exposing a shared join producer:\n{}",
        result.plan
    );
    assert_eq!(
        result
            .plan
            .nodes()
            .iter()
            .filter(|node| {
                let PlanNode::Join(join) = &node.kind else { return false };
                subtree_contains_gpu_source(&result.plan, join.inputs[0], "sales")
                    && subtree_contains_gpu_source(&result.plan, join.inputs[1], "returns")
            })
            .count(),
        1,
        "shared projection input should not be duplicated into multiple sales/returns joins:\n{}",
        result.plan
    );
}

#[test]
fn join_filter_reassociation_pushes_scalar_range_residuals_into_dimension_join_input() {
    let dir = unique_contract_temp_dir("join-filter-scalar-range");
    let date_path = dir.join("date_dim.parquet");
    let date_schema = schema(&["d_date_sk", "d_month_seq", "d_moy", "d_year"]);
    write_i32_record_batch_parquet(
        &date_path,
        Arc::clone(&date_schema),
        vec![
            Int32Array::from(vec![10, 11, 12, 13, 14]),
            Int32Array::from(vec![1187, 1188, 1189, 1190, 1191]),
            Int32Array::from(vec![12, 1, 2, 3, 4]),
            Int32Array::from(vec![1998, 1999, 1999, 1999, 1999]),
        ],
        8,
    );
    let plan = scalar_range_residual_plan(&date_path, Arc::clone(&date_schema));

    let result = optimize_query_plan(plan, &NativeOptimizerOptions::default(), rapids_analysis_context());

    assert!(
        result.diagnostics.join_filter_reassociation.applied_count >= 2,
        "lower and upper scalar residuals should both push necessary filters"
    );
    assert!(
        result.plan.nodes().iter().any(|node| {
            matches!(
                &node.kind,
                PlanNode::Filter(filter)
                    if filter.predicate
                        == Expr::binary(
                            Expr::column("d_month_seq"),
                            BinaryOp::GtEq,
                            Expr::literal(LiteralValue::Int32(1188))
                        )
            )
        }),
        "lower scalar bound should be replayed as a dimension-side filter"
    );
    assert!(
        result.plan.nodes().iter().any(|node| {
            matches!(
                &node.kind,
                PlanNode::Filter(filter)
                    if filter.predicate
                        == Expr::binary(
                            Expr::column("d_month_seq"),
                            BinaryOp::LtEq,
                            Expr::literal(LiteralValue::Int32(1190))
                        )
            )
        }),
        "upper scalar bound should be replayed as a dimension-side filter"
    );
    assert!(
        result.plan.dynamic_filter_routes().iter().any(|route| {
            matches!(route.consumer, DynamicFilterConsumer::LocalParquetSource { .. })
                && route.producer_columns == ["d_date_sk".to_string()]
                && route.consumer_columns == ["ss_sold_date_sk".to_string()]
        }),
        "the filtered date build side should still produce a store-sales dynamic filter"
    );

    fs::remove_dir_all(dir).expect("temporary test directory should be removable");
}

#[test]
fn join_filter_reassociation_uses_date32_literals_for_temporal_scalar_range_residuals() {
    let dir = unique_contract_temp_dir("join-filter-date32-scalar-range");
    let date_path = dir.join("date_dim.parquet");
    let date_schema = Arc::new(Schema::new(vec![Field::new("d_date", DataType::Date32, false)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&date_schema),
        vec![Arc::new(arrow::array::Date32Array::from(vec![100, 110, 120])) as ArrayRef],
    )
    .expect("test batch should be valid");
    write_record_batch_parquet(&date_path, Arc::clone(&date_schema), &batch, 8);
    let plan = date32_scalar_range_residual_plan(&date_path, Arc::clone(&date_schema));

    let result = optimize_query_plan(plan, &NativeOptimizerOptions::default(), rapids_analysis_context());

    let expected = Expr::binary(
        Expr::binary(
            Expr::column("d_date"),
            BinaryOp::Add,
            Expr::literal(LiteralValue::IntervalMonthDayNano { months: 0, days: 5, nanoseconds: 0 }),
        ),
        BinaryOp::Lt,
        Expr::literal(LiteralValue::Date32(120)),
    );
    assert!(
        result.plan.nodes().iter().any(|node| {
            matches!(
                &node.kind,
                PlanNode::Filter(filter) if filter.predicate == expected
            )
        }),
        "temporal residual pushdown must keep the generated bound literal Date32-compatible"
    );

    fs::remove_dir_all(dir).expect("temporary test directory should be removable");
}

#[test]
fn join_filter_reassociation_declines_out_of_range_float_scalar_bounds() {
    let dir = unique_contract_temp_dir("join-filter-float-scalar-range");
    let scalar_path = dir.join("scalar.parquet");
    let scalar_schema = Arc::new(Schema::new(vec![Field::new("bound", DataType::Float64, false)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&scalar_schema),
        vec![Arc::new(arrow::array::Float64Array::from(vec![1.0e20])) as ArrayRef],
    )
    .expect("test batch should be valid");
    write_record_batch_parquet(&scalar_path, Arc::clone(&scalar_schema), &batch, 8);
    let plan = out_of_range_float_scalar_range_residual_plan(&scalar_path, Arc::clone(&scalar_schema));

    let result = optimize_query_plan(plan, &NativeOptimizerOptions::default(), rapids_analysis_context());

    assert_eq!(
        result.diagnostics.join_filter_reassociation.applied_count, 0,
        "out-of-range float values must not be saturated into integer residual bounds"
    );
    assert!(
        !result.plan.nodes().iter().any(|node| matches!(&node.kind, PlanNode::Filter(_))),
        "no residual-derived filter should be pushed from an out-of-range float scalar"
    );

    fs::remove_dir_all(dir).expect("temporary test directory should be removable");
}

fn inner_join_filter_over_projected_inner_join_plan() -> nexus_query_engine::plan::QueryPlan {
    let (builder, filtered) = inner_join_filter_over_projected_inner_join_builder(None, None);
    builder.finish_gpu_resident(filtered)
}

fn inner_join_filter_over_projected_inner_join_builder(
    parent_advisory: Option<AdvisoryOutputStatistics>,
    crossed_advisory: Option<AdvisoryOutputStatistics>,
) -> (QueryPlanBuilder, NodeId) {
    let mut builder = QueryPlanBuilder::new();
    let sales = builder.add_source(gpu_source("sales", &["s_item", "s_ticket", "customer", "amount"]));
    let returns = builder.add_source(gpu_source("returns", &["r_item", "r_ticket", "reason_key", "returned"]));
    let reason = builder.add_source(gpu_source("reason", &["reason_key"]));
    let sales_returns_schema =
        schema(&["s_item", "s_ticket", "customer", "amount", "r_item", "r_ticket", "reason_key", "returned"]);
    let sales_returns = builder.add_join(
        sales,
        returns,
        JoinSpec {
            advisory_output_statistics: crossed_advisory,
            ..JoinSpec::new(
            JoinType::Inner,
            vec![
                JoinKey::new(ColumnRef::input(0, "s_item", "s_item"), ColumnRef::input(1, "r_item", "r_item")),
                JoinKey::new(ColumnRef::input(0, "s_ticket", "s_ticket"), ColumnRef::input(1, "r_ticket", "r_ticket")),
            ],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "s_item", "s_item"), "s_item"),
            JoinOutputColumn::left(ColumnRef::input(0, "s_ticket", "s_ticket"), "s_ticket"),
            JoinOutputColumn::left(ColumnRef::input(0, "customer", "customer"), "customer"),
            JoinOutputColumn::left(ColumnRef::input(0, "amount", "amount"), "amount"),
            JoinOutputColumn::right(ColumnRef::input(1, "r_item", "r_item"), "r_item"),
            JoinOutputColumn::right(ColumnRef::input(1, "r_ticket", "r_ticket"), "r_ticket"),
            JoinOutputColumn::right(ColumnRef::input(1, "reason_key", "reason_key"), "reason_key"),
            JoinOutputColumn::right(ColumnRef::input(1, "returned", "returned"), "returned"),
        ])
        },
        sales_returns_schema,
    );
    let projected_schema = schema(&["customer", "net_amount", "reason_key"]);
    let projected = builder.add_projection(
        sales_returns,
        vec![
            ProjectionOutput::new(Expr::column("customer"), Field::new("customer", DataType::Int32, false)),
            ProjectionOutput::new(
                Expr::binary(Expr::column("amount"), BinaryOp::Subtract, Expr::column("returned")),
                Field::new("net_amount", DataType::Int32, false),
            ),
            ProjectionOutput::new(Expr::column("reason_key"), Field::new("reason_key", DataType::Int32, false)),
        ],
    );
    let filtered = builder.add_join(
        projected,
        reason,
        JoinSpec {
            advisory_output_statistics: parent_advisory,
            ..JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "reason_key", "reason_key"),
                ColumnRef::input(1, "reason_key", "reason_key"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "customer", "customer"), "customer"),
            JoinOutputColumn::left(ColumnRef::input(0, "net_amount", "net_amount"), "net_amount"),
        ])
        },
        schema(&["customer", "net_amount"]),
    );
    assert_eq!(projected_schema.fields().len(), 3);
    (builder, filtered)
}

fn nested_inner_join_filter_plan() -> nexus_query_engine::plan::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let sales = builder.add_source(gpu_source("sales", &["sold_date_sk", "customer_sk", "amount"]));
    let customer = builder.add_source(gpu_source("customer", &["customer_sk", "addr_sk"]));
    let address = builder.add_source(gpu_source("address", &["addr_sk", "zip"]));
    let date = builder.add_source(gpu_source("date", &["date_sk"]));

    let sales_customer = builder.add_join(
        sales,
        customer,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "customer_sk", "customer_sk"),
                ColumnRef::input(1, "customer_sk", "customer_sk"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "sold_date_sk", "sold_date_sk"), "sold_date_sk"),
            JoinOutputColumn::left(ColumnRef::input(0, "amount", "amount"), "amount"),
            JoinOutputColumn::right(ColumnRef::input(1, "addr_sk", "addr_sk"), "addr_sk"),
        ]),
        schema(&["sold_date_sk", "amount", "addr_sk"]),
    );
    let projected_sales_customer = builder.add_projection(
        sales_customer,
        vec![
            ProjectionOutput::new(Expr::column("sold_date_sk"), Field::new("sold_date_sk", DataType::Int32, false)),
            ProjectionOutput::new(Expr::column("amount"), Field::new("amount", DataType::Int32, false)),
            ProjectionOutput::new(Expr::column("addr_sk"), Field::new("addr_sk", DataType::Int32, false)),
        ],
    );
    let address_sales = builder.add_join(
        address,
        projected_sales_customer,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "addr_sk", "addr_sk"), ColumnRef::input(1, "addr_sk", "addr_sk"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::right(ColumnRef::input(1, "sold_date_sk", "sold_date_sk"), "sold_date_sk"),
            JoinOutputColumn::right(ColumnRef::input(1, "amount", "amount"), "amount"),
            JoinOutputColumn::left(ColumnRef::input(0, "zip", "zip"), "zip"),
        ]),
        schema(&["sold_date_sk", "amount", "zip"]),
    );
    let date_filtered = builder.add_join(
        date,
        address_sales,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "date_sk", "date_sk"),
                ColumnRef::input(1, "sold_date_sk", "sold_date_sk"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::right(ColumnRef::input(1, "amount", "amount"), "amount"),
            JoinOutputColumn::right(ColumnRef::input(1, "zip", "zip"), "zip"),
        ]),
        schema(&["amount", "zip"]),
    );
    builder.finish_gpu_resident(date_filtered)
}

fn projected_parent_join_filter_over_fact_join_plan() -> nexus_query_engine::plan::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let supplier = builder.add_source(gpu_source("supplier", &["s_suppkey", "s_nationkey"]));
    let partsupp = builder.add_source(gpu_source("partsupp", &["ps_suppkey", "ps_partkey", "ps_supplycost"]));
    let nation = builder.add_source(gpu_source("nation", &["n_nationkey", "n_regionkey"]));
    let region = builder.add_source(gpu_source("region", &["r_regionkey"]));

    let filtered_nation = builder.add_join(
        nation,
        region,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "n_regionkey", "n_regionkey"),
                ColumnRef::input(1, "r_regionkey", "r_regionkey"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "n_nationkey", "n_nationkey"), "n_nationkey"),
            JoinOutputColumn::left(ColumnRef::input(0, "n_regionkey", "n_regionkey"), "n_regionkey"),
        ])
        .with_build_side(JoinSide::Right),
        schema(&["n_nationkey", "n_regionkey"]),
    );
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
            JoinOutputColumn::right(ColumnRef::input(1, "ps_suppkey", "ps_suppkey"), "ps_suppkey"),
            JoinOutputColumn::right(ColumnRef::input(1, "ps_partkey", "ps_partkey"), "ps_partkey"),
            JoinOutputColumn::right(ColumnRef::input(1, "ps_supplycost", "ps_supplycost"), "ps_supplycost"),
            JoinOutputColumn::left(ColumnRef::input(0, "s_nationkey", "s_nationkey"), "s_nationkey"),
        ])
        .with_build_side(JoinSide::Right),
        schema(&["ps_suppkey", "ps_partkey", "ps_supplycost", "s_nationkey"]),
    );
    let parent = builder.add_join(
        filtered_nation,
        supplier_partsupp,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "n_nationkey", "n_nationkey"),
                ColumnRef::input(1, "s_nationkey", "s_nationkey"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::right(ColumnRef::input(1, "ps_partkey", "ps_partkey"), "ps_partkey"),
            JoinOutputColumn::right(ColumnRef::input(1, "ps_supplycost", "ps_supplycost"), "ps_supplycost"),
            JoinOutputColumn::left(ColumnRef::input(0, "n_regionkey", "n_regionkey"), "n_regionkey"),
        ])
        .with_build_side(JoinSide::Right),
        schema(&["ps_partkey", "ps_supplycost", "n_regionkey"]),
    );
    let projected = builder.add_projection(
        parent,
        vec![
            ProjectionOutput::new(Expr::column("ps_partkey"), Field::new("ps_partkey", DataType::Int32, false)),
            ProjectionOutput::new(Expr::column("ps_supplycost"), Field::new("ps_supplycost", DataType::Int32, false)),
        ],
    );
    builder.finish_gpu_resident(projected)
}

fn shared_projected_parent_join_filter_plan() -> nexus_query_engine::plan::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let sales = builder.add_source(gpu_source("sales", &["s_item", "s_ticket", "customer", "amount"]));
    let returns = builder.add_source(gpu_source("returns", &["r_item", "r_ticket", "reason_key", "returned"]));
    let reason = builder.add_source(gpu_source("reason", &["reason_key", "reason_group"]));

    let sales_returns = builder.add_join(
        sales,
        returns,
        JoinSpec::new(
            JoinType::Inner,
            vec![
                JoinKey::new(ColumnRef::input(0, "s_item", "s_item"), ColumnRef::input(1, "r_item", "r_item")),
                JoinKey::new(ColumnRef::input(0, "s_ticket", "s_ticket"), ColumnRef::input(1, "r_ticket", "r_ticket")),
            ],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "s_item", "s_item"), "s_item"),
            JoinOutputColumn::left(ColumnRef::input(0, "s_ticket", "s_ticket"), "s_ticket"),
            JoinOutputColumn::left(ColumnRef::input(0, "customer", "customer"), "customer"),
            JoinOutputColumn::left(ColumnRef::input(0, "amount", "amount"), "amount"),
            JoinOutputColumn::right(ColumnRef::input(1, "reason_key", "reason_key"), "reason_key"),
        ]),
        schema(&["s_item", "s_ticket", "customer", "amount", "reason_key"]),
    );
    let reason_join = builder.add_join(
        sales_returns,
        reason,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "reason_key", "reason_key"),
                ColumnRef::input(1, "reason_key", "reason_key"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "customer", "customer"), "customer"),
            JoinOutputColumn::left(ColumnRef::input(0, "amount", "amount"), "amount"),
            JoinOutputColumn::left(ColumnRef::input(0, "reason_key", "reason_key"), "reason_key"),
            JoinOutputColumn::right(ColumnRef::input(1, "reason_group", "reason_group"), "reason_group"),
        ]),
        schema(&["customer", "amount", "reason_key", "reason_group"]),
    );
    let shared_projected = builder.add_projection(
        reason_join,
        vec![
            ProjectionOutput::new(Expr::column("customer"), Field::new("customer", DataType::Int32, false)),
            ProjectionOutput::new(Expr::column("amount"), Field::new("amount", DataType::Int32, false)),
            ProjectionOutput::new(Expr::column("reason_key"), Field::new("reason_key", DataType::Int32, false)),
        ],
    );
    let union =
        builder.add_union(vec![shared_projected, shared_projected], schema(&["customer", "amount", "reason_key"]));
    builder.finish_gpu_resident(union)
}

fn subtree_contains_gpu_source(plan: &nexus_query_engine::plan::QueryPlan, node_id: NodeId, source_name: &str) -> bool {
    let Some(node) = plan.node(node_id) else {
        return false;
    };
    if let PlanNode::Source(source) = &node.kind
        && let SourceNode::GpuResident(source) = source.as_ref()
    {
        return source.input_name == source_name;
    }
    node.input_ids().iter().any(|input| subtree_contains_gpu_source(plan, *input, source_name))
}

fn scalar_range_residual_plan(date_path: &Path, date_schema: Arc<Schema>) -> nexus_query_engine::plan::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let sales = builder.add_source(estimated_local_source_with_fields(
        "scalar_range_store_sales",
        &["ss_sold_date_sk", "ss_ext_sales_price"],
        512 * 1024 * 1024,
        100_000_000,
    ));
    let date_source = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new(date_path)
                .with_row_group_rows(vec![5])
                .with_projected_row_group_bytes(vec![1024], vec![1024]),
        ],
        Arc::clone(&date_schema),
        vec!["d_date_sk".to_string(), "d_month_seq".to_string(), "d_moy".to_string(), "d_year".to_string()],
    )));

    let date_all = builder.add_projection(
        date_source,
        vec![
            ProjectionOutput::new(Expr::column("d_date_sk"), Field::new("d_date_sk", DataType::Int32, false)),
            ProjectionOutput::new(Expr::column("d_month_seq"), Field::new("d_month_seq", DataType::Int32, false)),
        ],
    );
    let sales_date = builder.add_join(
        sales,
        date_all,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "ss_sold_date_sk", "ss_sold_date_sk"),
                ColumnRef::input(1, "d_date_sk", "d_date_sk"),
            )],
        )
        .with_build_side(JoinSide::Right)
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "ss_ext_sales_price", "ss_ext_sales_price"), "price"),
            JoinOutputColumn::right(ColumnRef::input(1, "d_month_seq", "d_month_seq"), "d_month_seq"),
        ]),
        schema(&["price", "d_month_seq"]),
    );

    let december = builder.add_filter(
        date_source,
        Expr::binary(
            Expr::binary(Expr::column("d_moy"), BinaryOp::Eq, Expr::literal(LiteralValue::Int32(12))),
            BinaryOp::And,
            Expr::binary(Expr::column("d_year"), BinaryOp::Eq, Expr::literal(LiteralValue::Int32(1998))),
        ),
        Arc::clone(&date_schema),
    );
    let month_projection = builder.add_projection(
        december,
        vec![ProjectionOutput::new(Expr::column("d_month_seq"), Field::new("d_month_seq", DataType::Int32, false))],
    );
    let lower_projection = builder.add_projection(
        month_projection,
        vec![ProjectionOutput::new(
            Expr::binary(Expr::column("d_month_seq"), BinaryOp::Add, Expr::literal(LiteralValue::Int64(1)))
                .alias("lower_month"),
            Field::new("lower_month", DataType::Int64, false),
        )],
    );
    let lower = builder.add_aggregate(
        lower_projection,
        AggregateSpec::new(vec![Expr::column("lower_month")], vec![]),
        Arc::new(Schema::new(vec![Field::new("lower_month", DataType::Int64, false)])),
    );
    let lower_join = builder.add_join(
        sales_date,
        lower,
        JoinSpec::new(JoinType::Inner, vec![])
            .with_build_side(JoinSide::Right)
            .with_residual_predicate(Expr::binary(
                Expr::input_column(0, "d_month_seq", "d_month_seq"),
                BinaryOp::GtEq,
                Expr::input_column(1, "lower_month", "lower_month"),
            ))
            .with_output_projection(vec![
                JoinOutputColumn::left(ColumnRef::input(0, "price", "price"), "price"),
                JoinOutputColumn::left(ColumnRef::input(0, "d_month_seq", "d_month_seq"), "d_month_seq"),
            ]),
        schema(&["price", "d_month_seq"]),
    );

    let upper_projection = builder.add_projection(
        month_projection,
        vec![ProjectionOutput::new(
            Expr::binary(Expr::column("d_month_seq"), BinaryOp::Add, Expr::literal(LiteralValue::Int64(3)))
                .alias("upper_month"),
            Field::new("upper_month", DataType::Int64, false),
        )],
    );
    let upper = builder.add_aggregate(
        upper_projection,
        AggregateSpec::new(vec![Expr::column("upper_month")], vec![]),
        Arc::new(Schema::new(vec![Field::new("upper_month", DataType::Int64, false)])),
    );
    let upper_join = builder.add_join(
        lower_join,
        upper,
        JoinSpec::new(JoinType::Inner, vec![])
            .with_build_side(JoinSide::Right)
            .with_residual_predicate(Expr::binary(
                Expr::input_column(0, "d_month_seq", "d_month_seq"),
                BinaryOp::LtEq,
                Expr::input_column(1, "upper_month", "upper_month"),
            ))
            .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "price", "price"), "price")]),
        schema(&["price"]),
    );
    builder.finish_gpu_resident(upper_join)
}

fn date32_scalar_range_residual_plan(
    date_path: &Path,
    date_schema: Arc<Schema>,
) -> nexus_query_engine::plan::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let left_schema = Arc::new(Schema::new(vec![Field::new("d_date", DataType::Date32, false)]));
    let left =
        builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("left_dates", Arc::clone(&left_schema))));
    let right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new(date_path)
                .with_row_group_rows(vec![3])
                .with_projected_row_group_bytes(vec![1024], vec![1024]),
        ],
        date_schema,
        vec!["d_date".to_string()],
    )));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(JoinType::Inner, vec![])
            .with_build_side(JoinSide::Right)
            .with_residual_predicate(Expr::binary(
                Expr::input_column(1, "d_date", "d_date"),
                BinaryOp::Gt,
                Expr::binary(
                    Expr::input_column(0, "d_date", "d_date"),
                    BinaryOp::Add,
                    Expr::literal(LiteralValue::IntervalMonthDayNano { months: 0, days: 5, nanoseconds: 0 }),
                ),
            ))
            .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "d_date", "d_date"), "d_date")]),
        left_schema,
    );
    builder.finish_gpu_resident(join)
}

fn out_of_range_float_scalar_range_residual_plan(
    scalar_path: &Path,
    scalar_schema: Arc<Schema>,
) -> nexus_query_engine::plan::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(estimated_local_source_with_fields(
        "float_scalar_fact",
        &["value"],
        64 * 1024 * 1024,
        1_000_000,
    ));
    let right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new(scalar_path)
                .with_row_group_rows(vec![1])
                .with_projected_row_group_bytes(vec![1024], vec![1024]),
        ],
        scalar_schema,
        vec!["bound".to_string()],
    )));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(JoinType::Inner, vec![])
            .with_build_side(JoinSide::Right)
            .with_residual_predicate(Expr::binary(
                Expr::input_column(0, "value", "value"),
                BinaryOp::Lt,
                Expr::input_column(1, "bound", "bound"),
            ))
            .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "value", "value"), "value")]),
        schema(&["value"]),
    );
    builder.finish_gpu_resident(join)
}
