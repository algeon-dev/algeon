#[test]
fn join_filter_reassociation_pushes_dimension_filter_below_fact_join_without_losing_multiplicity() {
    let plan = inner_join_filter_over_projected_inner_join_plan();

    let result = optimize_query_plan(plan, &NativeOptimizerOptions::default());

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

    let result = optimize_query_plan(plan, &NativeOptimizerOptions::default());

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

fn inner_join_filter_over_projected_inner_join_plan() -> nexus_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let sales = builder.add_source(gpu_source("sales", &["s_item", "s_ticket", "customer", "amount"]));
    let returns = builder.add_source(gpu_source("returns", &["r_item", "r_ticket", "reason_key", "returned"]));
    let reason = builder.add_source(gpu_source("reason", &["reason_key"]));
    let sales_returns_schema =
        schema(&["s_item", "s_ticket", "customer", "amount", "r_item", "r_ticket", "reason_key", "returned"]);
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
            JoinOutputColumn::right(ColumnRef::input(1, "r_item", "r_item"), "r_item"),
            JoinOutputColumn::right(ColumnRef::input(1, "r_ticket", "r_ticket"), "r_ticket"),
            JoinOutputColumn::right(ColumnRef::input(1, "reason_key", "reason_key"), "reason_key"),
            JoinOutputColumn::right(ColumnRef::input(1, "returned", "returned"), "returned"),
        ]),
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
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "reason_key", "reason_key"),
                ColumnRef::input(1, "reason_key", "reason_key"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "customer", "customer"), "customer"),
            JoinOutputColumn::left(ColumnRef::input(0, "net_amount", "net_amount"), "net_amount"),
        ]),
        schema(&["customer", "net_amount"]),
    );
    assert_eq!(projected_schema.fields().len(), 3);
    builder.finish_gpu_resident(filtered)
}

fn nested_inner_join_filter_plan() -> nexus_query_engine::QueryPlan {
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
