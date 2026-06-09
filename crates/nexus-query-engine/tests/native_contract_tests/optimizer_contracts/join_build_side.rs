#[test]
fn native_optimizer_flips_inner_equi_join_to_build_smaller_estimated_left_side() {
    let plan = estimated_inner_equi_join_plan(16 * 1024 * 1024, 256 * 1024 * 1024);

    let result = optimize_query_plan(plan, &NativeOptimizerOptions::default());

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    let PlanNode::Join(join) = &result.plan.root_node().kind else {
        panic!("optimized root should be the inner join");
    };
    assert_eq!(join.spec.join_type, JoinType::Inner);
    assert_eq!(join.spec.build_side, JoinSide::Left);
}

#[test]
fn native_optimizer_flips_inner_keyed_residual_join_to_build_smaller_estimated_left_side() {
    let plan =
        estimated_inner_residual_join_plan(JoinSide::Right, 16 * 1024 * 1024, 100_000, 256 * 1024 * 1024, 10_000_000);

    let result = optimize_query_plan(plan, &NativeOptimizerOptions::default());

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    let PlanNode::Join(join) = &result.plan.root_node().kind else {
        panic!("optimized root should be the inner residual join");
    };
    assert_eq!(join.spec.join_type, JoinType::Inner);
    assert!(join.spec.residual_predicate.is_some());
    assert_eq!(join.spec.build_side, JoinSide::Left);
}

#[test]
fn native_optimizer_disabled_keeps_default_inner_equi_join_build_side() {
    let plan = estimated_inner_equi_join_plan(16 * 1024 * 1024, 256 * 1024 * 1024);

    let result = optimize_query_plan(plan, &NativeOptimizerOptions::disabled());

    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    let PlanNode::Join(join) = &result.plan.root_node().kind else {
        panic!("optimized root should be the inner join");
    };
    assert_eq!(join.spec.build_side, JoinSide::Right);
}

#[test]
fn join_build_side_selection_switches_inner_equi_join_to_small_left_input() {
    let plan = estimated_inner_join_plan(JoinSide::Right, 1_024, 100, 9 * 1024 * 1024 * 1024, 100_000_000);

    let result = optimize_query_plan(plan, &NativeOptimizerOptions::default());

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    assert!(result.diagnostics.semijoin_reduction.is_empty());
    assert_eq!(result.diagnostics.join_build_side_selection.considered_count, 1);
    assert_eq!(result.diagnostics.join_build_side_selection.applied_count, 1);
    let candidate = &result.diagnostics.join_build_side_selection.candidates[0];
    assert_eq!(candidate.before_build_side, JoinSide::Right);
    assert_eq!(candidate.after_build_side, Some(JoinSide::Left));
    assert!(candidate.applied);
    assert_eq!(candidate.left_estimated_bytes, Some(1_024));
    assert_eq!(candidate.right_estimated_bytes, Some(9 * 1024 * 1024 * 1024));

    let PlanNode::Join(join) = &result.plan.root_node().kind else {
        panic!("optimized root should remain an inner join");
    };
    assert_eq!(join.spec.build_side, JoinSide::Left);
}

#[test]
fn join_build_side_selection_prefers_grouped_aggregate_keyset_side_when_size_is_unknown() {
    let mut builder = QueryPlanBuilder::new();
    let raw_keyset =
        builder.add_source(estimated_local_source("returns", "raw_customer_sk", 512 * 1024 * 1024, 10_000_000));
    let aggregate = builder.add_aggregate(
        raw_keyset,
        AggregateSpec::new(vec![Expr::column("raw_customer_sk")], Vec::new()),
        schema(&["raw_customer_sk"]),
    );
    let projected = builder.add_projection(
        aggregate,
        vec![ProjectionOutput::new(
            Expr::column("raw_customer_sk"),
            Field::new("ctr_customer_sk", DataType::Int32, false),
        )],
    );
    let customer =
        builder.add_source(estimated_local_source("customer", "c_customer_sk", 256 * 1024 * 1024, 2_000_000));
    let join = builder.add_join(
        projected,
        customer,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "ctr_customer_sk", "ctr_customer_sk"),
                ColumnRef::input(1, "c_customer_sk", "c_customer_sk"),
            )],
        )
        .with_output_projection(vec![JoinOutputColumn::left(
            ColumnRef::input(0, "ctr_customer_sk", "ctr_customer_sk"),
            "ctr_customer_sk",
        )])
        .with_build_side(JoinSide::Right),
        schema(&["ctr_customer_sk"]),
    );
    let plan = builder.finish_gpu_resident(join);

    let result = optimize_query_plan(plan, &NativeOptimizerOptions::default());

    let candidate = result
        .diagnostics
        .join_build_side_selection
        .candidates
        .iter()
        .find(|candidate| candidate.candidate_node_id == join)
        .expect("join candidate should be considered");
    assert!(candidate.applied);
    assert_eq!(candidate.after_build_side, Some(JoinSide::Left));
    let PlanNode::Join(join) = &result.plan.root_node().kind else {
        panic!("optimized root should remain the join");
    };
    assert_eq!(join.spec.build_side, JoinSide::Left);
}

#[test]
fn join_build_side_selection_does_not_prefer_grouped_aggregate_side_when_join_key_is_measure() {
    let mut builder = QueryPlanBuilder::new();
    let sales = builder.add_source(estimated_sales_source_with_quarter_predicate(1, 512 * 1024 * 1024));
    let aggregate = builder.add_aggregate(
        sales,
        AggregateSpec::new(
            vec![Expr::column("county")],
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("sales"), "store_sales")],
        ),
        schema(&["county", "store_sales"]),
    );
    let customer =
        builder.add_source(estimated_local_source("customer", "c_customer_sk", 256 * 1024 * 1024, 2_000_000));
    let join = builder.add_join(
        aggregate,
        customer,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "store_sales", "store_sales"),
                ColumnRef::input(1, "c_customer_sk", "c_customer_sk"),
            )],
        )
        .with_output_projection(vec![JoinOutputColumn::right(
            ColumnRef::input(1, "c_customer_sk", "c_customer_sk"),
            "c_customer_sk",
        )])
        .with_build_side(JoinSide::Right),
        schema(&["c_customer_sk"]),
    );
    let plan = builder.finish_gpu_resident(join);

    let result = optimize_query_plan(plan, &NativeOptimizerOptions::default());

    let candidate = result
        .diagnostics
        .join_build_side_selection
        .candidates
        .iter()
        .find(|candidate| candidate.candidate_node_id == join)
        .expect("join candidate should be considered");
    assert!(!candidate.applied);
    assert_eq!(candidate.after_build_side, None);
    let PlanNode::Join(join) = &result.plan.root_node().kind else {
        panic!("optimized root should remain the join");
    };
    assert_eq!(join.spec.build_side, JoinSide::Right);
}

#[test]
fn disabled_native_optimizer_keeps_join_build_side_selection_off() {
    let plan = estimated_inner_join_plan(JoinSide::Right, 1_024, 100, 9 * 1024 * 1024 * 1024, 100_000_000);

    let result = optimize_query_plan(plan.clone(), &NativeOptimizerOptions::disabled());

    assert_eq!(result.plan, plan);
    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    assert!(result.diagnostics.is_empty());
}

fn estimated_inner_equi_join_plan(left_bytes: u64, right_bytes: u64) -> nexus_query_engine::QueryPlan {
    estimated_inner_join_plan(JoinSide::Right, left_bytes, 1, right_bytes, 1)
}

fn estimated_inner_join_plan(
    build_side: JoinSide,
    left_bytes: u64,
    left_rows: u64,
    right_bytes: u64,
    right_rows: u64,
) -> nexus_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(estimated_local_source("left", "left_key", left_bytes, left_rows));
    let right = builder.add_source(estimated_local_source("right", "right_key", right_bytes, right_rows));
    let output_schema = schema(&["left_key", "right_key"]);
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "left_key", "left_key"),
                ColumnRef::input(1, "right_key", "right_key"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "left_key", "left_key"), "left_key"),
            JoinOutputColumn::right(ColumnRef::input(1, "right_key", "right_key"), "right_key"),
        ])
        .with_build_side(build_side),
        output_schema,
    );
    builder.finish_gpu_resident(join)
}

fn estimated_inner_residual_join_plan(
    build_side: JoinSide,
    left_bytes: u64,
    left_rows: u64,
    right_bytes: u64,
    right_rows: u64,
) -> nexus_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(estimated_local_source_with_fields(
        "left_residual",
        &["left_key", "left_payload"],
        left_bytes,
        left_rows,
    ));
    let right = builder.add_source(estimated_local_source_with_fields(
        "right_residual",
        &["right_key", "right_payload"],
        right_bytes,
        right_rows,
    ));
    let output_schema = schema(&["left_key", "right_key"]);
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "left_key", "left_key"),
                ColumnRef::input(1, "right_key", "right_key"),
            )],
        )
        .with_residual_predicate(Expr::binary(
            Expr::input_column(0, "left_payload", "left_payload"),
            BinaryOp::Gt,
            Expr::input_column(1, "right_payload", "right_payload"),
        ))
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "left_key", "left_key"), "left_key"),
            JoinOutputColumn::right(ColumnRef::input(1, "right_key", "right_key"), "right_key"),
        ])
        .with_build_side(build_side),
        output_schema,
    );
    builder.finish_gpu_resident(join)
}
