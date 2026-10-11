use cudf::testing::run_named_cases;

use super::*;

#[test]
fn outer_join_null_rejection_converts_left_join_when_inner_join_key_rejects_right_nulls() {
    let plan = inner_join_over_projected_left_join_right_key_plan();
    let options = NativeOptimizerOptions {
        join_filter_reassociation: JoinFilterReassociationOptions::disabled(),
        ..NativeOptimizerOptions::default()
    };

    let result = optimize_query_plan(
        plan,
        &options,
        rapids_analysis_context(PlanningPolicy::default()),
        local_dimension_pruning_resolver(),
    )
    .unwrap();

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    assert_eq!(result.diagnostics.outer_join_null_rejection.considered_count, 1);
    assert_eq!(result.diagnostics.outer_join_null_rejection.applied_count, 1);
    assert!(result.diagnostics.outer_join_null_rejection.candidates[0].applied);
    assert_eq!(result.diagnostics.outer_join_null_rejection.candidates[0].rejected_reason, None);

    let PlanNode::Join(root_join) = &result.plan.root_node().kind else {
        panic!("root should remain an inner join");
    };
    let PlanNode::Projection(projection) = &result.plan.node(root_join.inputs[0]).expect("projection").kind else {
        panic!("root left input should remain a projection");
    };
    let PlanNode::Join(rewritten_join) = &result.plan.node(projection.input).expect("left join").kind else {
        panic!("projection input should be rewritten join");
    };
    assert_eq!(rewritten_join.spec.join_type, JoinType::Inner);
}

#[test]
fn outer_join_null_rejection_preserves_left_join_build_side_after_build_side_selection() {
    let plan =
        inner_join_over_estimated_projected_left_join_right_key_plan(1024 * 1024, 10_000, 32 * 1024 * 1024, 1_000_000);
    let options = NativeOptimizerOptions {
        join_build_side_selection: JoinBuildSideSelectionOptions::default(),
        outer_join_null_rejection: OuterJoinNullRejectionOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(
        plan,
        &options,
        rapids_analysis_context(PlanningPolicy::default()),
        local_dimension_pruning_resolver(),
    )
    .unwrap();

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    assert_eq!(result.diagnostics.outer_join_null_rejection.considered_count, 1);
    assert_eq!(result.diagnostics.outer_join_null_rejection.applied_count, 1);
    let rewritten_id = result.diagnostics.outer_join_null_rejection.candidates[0].candidate_node_id;
    assert!(
        result
            .diagnostics
            .join_build_side_selection
            .candidates
            .iter()
            .all(|candidate| candidate.candidate_node_id != rewritten_id),
        "build-side selection must not revisit the newly inner join after null rejection"
    );
    let PlanNode::Join(rewritten_join) = &result.plan.node(rewritten_id).expect("rewritten join").kind else {
        panic!("candidate should remain a join node");
    };
    assert_eq!(rewritten_join.spec.join_type, JoinType::Inner);
    assert_eq!(rewritten_join.spec.build_side, JoinSide::Right);
}

fn outer_join_null_rejection_keeps_left_join_when_key_comes_from_preserved_side() {
    let plan = inner_join_over_projected_left_join_left_key_plan();

    let result = optimize_query_plan(
        plan.clone(),
        &outer_join_null_rejection_only_options(),
        rapids_analysis_context(PlanningPolicy::default()),
        local_dimension_pruning_resolver(),
    )
    .unwrap();

    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    assert_eq!(result.plan, plan);
    assert!(result.diagnostics.outer_join_null_rejection.is_empty());
}

#[test]
fn t_outer_join_keeps_cases() {
    run_named_cases(&[
        (
            "optimizer_contracts::outer_join_null_rejection::outer_join_null_rejection_keeps_left_join_when_key_comes_from_preserved_side",
            outer_join_null_rejection_keeps_left_join_when_key_comes_from_preserved_side,
        ),
        (
            "optimizer_contracts::outer_join_null_rejection::outer_join_null_rejection_keeps_left_join_below_limit",
            outer_join_null_rejection_keeps_left_join_below_limit,
        ),
        (
            "optimizer_contracts::outer_join_null_rejection::outer_join_null_rejection_keeps_marker_null_filter_with_nulls_match_join",
            outer_join_null_rejection_keeps_marker_null_filter_with_nulls_match_join,
        ),
        (
            "optimizer_contracts::outer_join_null_rejection::outer_join_null_rejection_keeps_marker_null_filter_when_marker_escapes_projection",
            outer_join_null_rejection_keeps_marker_null_filter_when_marker_escapes_projection,
        ),
        (
            "optimizer_contracts::outer_join_null_rejection::outer_join_null_rejection_keeps_large_null_introducing_input",
            outer_join_null_rejection_keeps_large_null_introducing_input,
        ),
        (
            "optimizer_contracts::outer_join_null_rejection::outer_join_null_rejection_rejects_shared_outer_join_output",
            outer_join_null_rejection_rejects_shared_outer_join_output,
        ),
    ]);
}

fn outer_join_null_rejection_rejects_shared_outer_join_output() {
    let plan = shared_left_join_output_with_one_null_rejecting_consumer_plan();

    let result = optimize_query_plan(
        plan.clone(),
        &outer_join_null_rejection_only_options(),
        rapids_analysis_context(PlanningPolicy::default()),
        local_dimension_pruning_resolver(),
    )
    .unwrap();

    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    assert_eq!(result.plan, plan);
    assert_eq!(result.diagnostics.outer_join_null_rejection.considered_count, 1);
    assert_eq!(result.diagnostics.outer_join_null_rejection.applied_count, 0);
    assert_eq!(
        result.diagnostics.outer_join_null_rejection.candidates[0].rejected_reason,
        Some(OuterJoinNullRejectionRejectedReason::SharedOuterJoinOutput)
    );
}

fn outer_join_null_rejection_keeps_left_join_below_limit() {
    let plan = inner_join_over_limited_projected_left_join_right_key_plan();

    let result = optimize_query_plan(
        plan.clone(),
        &outer_join_null_rejection_only_options(),
        rapids_analysis_context(PlanningPolicy::default()),
        local_dimension_pruning_resolver(),
    )
    .unwrap();

    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    assert_eq!(result.plan, plan);
    assert_eq!(result.diagnostics.outer_join_null_rejection.applied_count, 0);
}

fn outer_join_null_rejection_keeps_large_null_introducing_input() {
    let plan = inner_join_over_large_left_join_right_key_plan();

    let result = optimize_query_plan(
        plan.clone(),
        &outer_join_null_rejection_only_options(),
        rapids_analysis_context(PlanningPolicy::default()),
        local_dimension_pruning_resolver(),
    )
    .unwrap();

    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    assert_eq!(result.plan, plan);
    assert_eq!(result.diagnostics.outer_join_null_rejection.considered_count, 1);
    assert_eq!(result.diagnostics.outer_join_null_rejection.applied_count, 0);
    assert_eq!(
        result.diagnostics.outer_join_null_rejection.candidates[0].rejected_reason,
        Some(OuterJoinNullRejectionRejectedReason::LargeNullIntroducingInput)
    );
}

#[test]
fn outer_join_null_rejection_revisits_large_input_after_aggregate_pushdown() {
    let plan = aggregate_over_reason_join_over_large_left_join_plan();
    let mut options = NativeOptimizerOptions::disabled();
    options.aggregate_join_pushdown = AggregateJoinPushdownOptions::default();
    options.outer_join_null_rejection = OuterJoinNullRejectionOptions::default();

    let result = optimize_query_plan(
        plan,
        &options,
        rapids_analysis_context(PlanningPolicy::default()),
        local_dimension_pruning_resolver(),
    )
    .unwrap();

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    assert_eq!(result.diagnostics.aggregate_join_pushdown.considered_count, 1);
    assert_eq!(result.diagnostics.aggregate_join_pushdown.applied_count, 1);
    assert_eq!(result.diagnostics.outer_join_null_rejection.considered_count, 2);
    assert_eq!(result.diagnostics.outer_join_null_rejection.applied_count, 1);
    assert_eq!(
        result.diagnostics.outer_join_null_rejection.candidates[0].rejected_reason,
        Some(OuterJoinNullRejectionRejectedReason::LargeNullIntroducingInput)
    );
    assert!(result.diagnostics.outer_join_null_rejection.candidates[1].applied);
    assert_eq!(result.diagnostics.outer_join_null_rejection.candidates[1].rejected_reason, None);
    assert_all_node_inputs_exist(&result.plan);
    assert!(
        result
            .plan
            .nodes()
            .iter()
            .filter_map(|node| match &node.kind {
                PlanNode::Join(join) => Some(join.spec.join_type.clone()),
                _ => None,
            })
            .all(|join_type| join_type != JoinType::Left),
        "post-pushdown grouped-key null rejection should convert the large left join"
    );
}

#[test]
fn outer_join_null_rejection_traces_rejected_key_through_union_and_group_key() {
    let plan = union_aggregate_over_left_join_right_key_plan();

    let result = optimize_query_plan(
        plan,
        &outer_join_null_rejection_only_options(),
        rapids_analysis_context(PlanningPolicy::default()),
        local_dimension_pruning_resolver(),
    )
    .unwrap();

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    assert_eq!(result.diagnostics.outer_join_null_rejection.considered_count, 1);
    assert_eq!(result.diagnostics.outer_join_null_rejection.applied_count, 1);
    assert!(result.diagnostics.outer_join_null_rejection.candidates[0].applied);
    assert!(
        result
            .plan
            .nodes()
            .iter()
            .filter_map(|node| match &node.kind {
                PlanNode::Join(join) => Some(join.spec.join_type.clone()),
                _ => None,
            })
            .all(|join_type| join_type != JoinType::Left),
        "the null-rejected left join should be converted to inner"
    );
}

#[test]
fn outer_join_null_rejection_converts_marker_null_filter_to_left_anti() {
    let plan = marker_null_filter_left_join_plan(MarkerNullFilterPlanVariant::Eligible);

    let result = optimize_query_plan(
        plan,
        &outer_join_null_rejection_only_options(),
        rapids_analysis_context(PlanningPolicy::default()),
        local_dimension_pruning_resolver(),
    )
    .unwrap();

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    assert_eq!(result.diagnostics.outer_join_null_rejection.considered_count, 1);
    assert_eq!(result.diagnostics.outer_join_null_rejection.applied_count, 1);

    let PlanNode::Projection(projection) = &result.plan.root_node().kind else {
        panic!("root should remain the final projection");
    };
    let PlanNode::Join(join) = &result.plan.node(projection.input).expect("anti join").kind else {
        panic!("projection should consume rewritten anti join");
    };
    assert_eq!(join.spec.join_type, JoinType::LEFT_ANTI);
    assert_eq!(join.spec.build_side, JoinSide::Right);
    assert!(join.spec.output_projection.iter().all(|column| column.side == JoinSide::Left));
    assert_eq!(join.spec.output_projection.len(), 2);
}

#[test]
fn outer_join_null_rejection_converts_marker_null_filter_when_key_schema_is_nullable() {
    let plan = marker_null_filter_left_join_plan(MarkerNullFilterPlanVariant::NullableMarker);

    let result = optimize_query_plan(
        plan,
        &outer_join_null_rejection_only_options(),
        rapids_analysis_context(PlanningPolicy::default()),
        local_dimension_pruning_resolver(),
    )
    .unwrap();

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    let PlanNode::Projection(projection) = &result.plan.root_node().kind else {
        panic!("root should remain the final projection");
    };
    let PlanNode::Join(join) = &result.plan.node(projection.input).expect("anti join").kind else {
        panic!("projection should consume rewritten anti join");
    };
    assert_eq!(join.spec.join_type, JoinType::LEFT_ANTI);
}

fn outer_join_null_rejection_keeps_marker_null_filter_with_nulls_match_join() {
    let plan = marker_null_filter_left_join_plan(MarkerNullFilterPlanVariant::NullsMatch);

    let result = optimize_query_plan(
        plan.clone(),
        &outer_join_null_rejection_only_options(),
        rapids_analysis_context(PlanningPolicy::default()),
        local_dimension_pruning_resolver(),
    )
    .unwrap();

    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    assert_eq!(result.plan, plan);
    assert!(result.diagnostics.outer_join_null_rejection.is_empty());
}

fn outer_join_null_rejection_keeps_marker_null_filter_when_marker_escapes_projection() {
    let plan = marker_null_filter_left_join_plan(MarkerNullFilterPlanVariant::MarkerEscapes);

    let result = optimize_query_plan(
        plan.clone(),
        &outer_join_null_rejection_only_options(),
        rapids_analysis_context(PlanningPolicy::default()),
        local_dimension_pruning_resolver(),
    )
    .unwrap();

    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    assert_eq!(result.plan, plan);
    assert!(result.diagnostics.outer_join_null_rejection.is_empty());
}

fn inner_join_over_projected_left_join_right_key_plan() -> QueryPlan {
    inner_join_over_projected_left_join_key_plan("r_key")
}

fn inner_join_over_projected_left_join_left_key_plan() -> QueryPlan {
    inner_join_over_projected_left_join_key_plan("l_key")
}

fn inner_join_over_projected_left_join_key_plan(rejecting_key: &str) -> QueryPlan {
    inner_join_over_projected_left_join_key_plan_with_sources(
        rejecting_key,
        gpu_source("left", &["l_key", "payload"]),
        gpu_source("right", &["r_key", "r_reason"]),
        None,
    )
}

fn inner_join_over_limited_projected_left_join_right_key_plan() -> QueryPlan {
    inner_join_over_projected_left_join_key_plan_with_sources(
        "r_key",
        gpu_source("left", &["l_key", "payload"]),
        gpu_source("right", &["r_key", "r_reason"]),
        Some(10),
    )
}

fn inner_join_over_estimated_projected_left_join_right_key_plan(
    left_bytes: u64,
    left_rows: u64,
    right_bytes: u64,
    right_rows: u64,
) -> QueryPlan {
    inner_join_over_projected_left_join_key_plan_with_sources(
        "r_key",
        estimated_local_source_with_fields("left", &["l_key", "payload"], left_bytes, left_rows),
        estimated_local_source_with_fields("right", &["r_key", "r_reason"], right_bytes, right_rows),
        None,
    )
}

fn inner_join_over_projected_left_join_key_plan_with_sources(
    rejecting_key: &str,
    left_source: SourceNode,
    right_source: SourceNode,
    limit_fetch: Option<usize>,
) -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(left_source);
    let right = builder.add_source(right_source);
    let dim = builder.add_source(gpu_source("dim", &[rejecting_key]));
    let left_join_schema = schema(&["l_key", "payload", "r_key", "r_reason"]);
    let left_join = builder
        .add_join(
            left,
            right,
            JoinSpec::new(
                JoinType::Left,
                vec![JoinKey::new(ColumnRef::input(0, "l_key", "l_key"), ColumnRef::input(1, "r_key", "r_key"))],
            )
            .with_output_projection(vec![
                JoinOutputColumn::left(ColumnRef::input(0, "l_key", "l_key"), "l_key"),
                JoinOutputColumn::left(ColumnRef::input(0, "payload", "payload"), "payload"),
                JoinOutputColumn::right(ColumnRef::input(1, "r_key", "r_key"), "r_key"),
                JoinOutputColumn::right(ColumnRef::input(1, "r_reason", "r_reason"), "r_reason"),
            ]),
            Arc::clone(&left_join_schema),
        )
        .unwrap();
    let projected = builder
        .add_projection(
            left_join,
            vec![
                ProjectionOutput::new(Expr::column("l_key"), Field::new("l_key", DataType::Int32, false)),
                ProjectionOutput::new(Expr::column("payload"), Field::new("payload", DataType::Int32, false)),
                ProjectionOutput::new(Expr::column("r_key"), Field::new("r_key", DataType::Int32, true)),
                ProjectionOutput::new(Expr::column("r_reason"), Field::new("r_reason", DataType::Int32, true)),
            ],
        )
        .unwrap();
    let root_schema = Arc::new(Schema::new(vec![
        Field::new("l_key", DataType::Int32, false),
        Field::new("payload", DataType::Int32, false),
        Field::new("r_key", DataType::Int32, true),
        Field::new("r_reason", DataType::Int32, true),
    ]));
    let root_input =
        limit_fetch.map_or(projected, |fetch| builder.add_limit(projected, LimitSpec::fetch(fetch)).unwrap());
    let root = builder
        .add_join(
            root_input,
            dim,
            JoinSpec::new(
                JoinType::Inner,
                vec![JoinKey::new(
                    ColumnRef::input(0, rejecting_key, rejecting_key),
                    ColumnRef::input(1, rejecting_key, rejecting_key),
                )],
            )
            .with_output_projection(left_projection(&["l_key", "payload", "r_key", "r_reason"])),
            root_schema,
        )
        .unwrap();
    builder.finish_gpu_resident(root).unwrap()
}

fn shared_left_join_output_with_one_null_rejecting_consumer_plan() -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(gpu_source("left", &["l_key"]));
    let right = builder.add_source(gpu_source("right", &["r_key"]));
    let dim = builder.add_source(gpu_source("dim", &["r_key"]));
    let left_join_schema = schema(&["l_key", "r_key"]);
    let left_join = builder
        .add_join(
            left,
            right,
            JoinSpec::new(
                JoinType::Left,
                vec![JoinKey::new(ColumnRef::input(0, "l_key", "l_key"), ColumnRef::input(1, "r_key", "r_key"))],
            )
            .with_output_projection(vec![
                JoinOutputColumn::left(ColumnRef::input(0, "l_key", "l_key"), "l_key"),
                JoinOutputColumn::right(ColumnRef::input(1, "r_key", "r_key"), "r_key"),
            ]),
            Arc::clone(&left_join_schema),
        )
        .unwrap();
    let rejecting = builder
        .add_join(
            left_join,
            dim,
            JoinSpec::new(
                JoinType::Inner,
                vec![JoinKey::new(ColumnRef::input(0, "r_key", "r_key"), ColumnRef::input(1, "r_key", "r_key"))],
            )
            .with_output_projection(left_projection(&["l_key", "r_key"])),
            Arc::clone(&left_join_schema),
        )
        .unwrap();
    let root = builder
        .add_join(
            left_join,
            rejecting,
            JoinSpec::new(
                JoinType::Inner,
                vec![JoinKey::new(ColumnRef::input(0, "l_key", "l_key"), ColumnRef::input(1, "l_key", "l_key"))],
            )
            .with_output_projection(left_projection(&["l_key", "r_key"])),
            left_join_schema,
        )
        .unwrap();
    builder.finish_gpu_resident(root).unwrap()
}

fn inner_join_over_large_left_join_right_key_plan() -> QueryPlan {
    inner_join_over_projected_left_join_key_plan_with_sources(
        "r_key",
        gpu_source("left", &["l_key", "payload"]),
        estimated_local_source_with_fields("large_right", &["r_key", "r_reason"], 1024 * 1024 * 1024, 10_000_000),
        None,
    )
}

fn aggregate_over_reason_join_over_large_left_join_plan() -> QueryPlan {
    let sales_schema = schema(&["ss_item_sk", "ss_customer_sk", "ss_ticket_number", "ss_quantity", "ss_sales_price"]);
    let returns_schema = schema(&["sr_item_sk", "sr_reason_sk", "sr_ticket_number", "sr_return_quantity"]);
    let left_join_schema = Arc::new(Schema::new(vec![
        Field::new("ss_customer_sk", DataType::Int32, false),
        Field::new("ss_quantity", DataType::Int32, false),
        Field::new("ss_sales_price", DataType::Int32, false),
        Field::new("sr_reason_sk", DataType::Int32, true),
        Field::new("sr_return_quantity", DataType::Int32, true),
    ]));
    let reason_join_schema = Arc::new(Schema::new(vec![
        Field::new("ss_customer_sk", DataType::Int32, false),
        Field::new("sr_return_quantity", DataType::Int32, true),
        Field::new("ss_quantity", DataType::Int32, false),
        Field::new("ss_sales_price", DataType::Int32, false),
    ]));
    let aggregate_schema = schema(&["ss_customer_sk", "sum_act_sales"]);
    let projection_schema = schema(&["ss_customer_sk", "sumsales"]);

    let mut builder = QueryPlanBuilder::new();
    let sales = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("store_sales", sales_schema)));
    let returns = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new("/tmp/store_returns.parquet")
                .with_projected_row_group_bytes(vec![1024 * 1024 * 1024], vec![1024 * 1024 * 1024])
                .with_row_group_rows(vec![10_000_000]),
        ],
        returns_schema,
        vec!["sr_item_sk".into(), "sr_reason_sk".into(), "sr_ticket_number".into(), "sr_return_quantity".into()],
    )));
    let reason = builder.add_source(gpu_source("reason", &["r_reason_sk"]));
    let left_join = builder
        .add_join(
            sales,
            returns,
            JoinSpec::new(
                JoinType::Left,
                vec![
                    JoinKey::new(
                        ColumnRef::input(0, "ss_item_sk", "ss_item_sk"),
                        ColumnRef::input(1, "sr_item_sk", "sr_item_sk"),
                    ),
                    JoinKey::new(
                        ColumnRef::input(0, "ss_ticket_number", "ss_ticket_number"),
                        ColumnRef::input(1, "sr_ticket_number", "sr_ticket_number"),
                    ),
                ],
            )
            .with_output_projection(vec![
                JoinOutputColumn::left(ColumnRef::input(0, "ss_customer_sk", "ss_customer_sk"), "ss_customer_sk"),
                JoinOutputColumn::left(ColumnRef::input(0, "ss_quantity", "ss_quantity"), "ss_quantity"),
                JoinOutputColumn::left(ColumnRef::input(0, "ss_sales_price", "ss_sales_price"), "ss_sales_price"),
                JoinOutputColumn::right(ColumnRef::input(1, "sr_reason_sk", "sr_reason_sk"), "sr_reason_sk"),
                JoinOutputColumn::right(
                    ColumnRef::input(1, "sr_return_quantity", "sr_return_quantity"),
                    "sr_return_quantity",
                ),
            ])
            .with_build_side(JoinSide::Right),
            left_join_schema,
        )
        .unwrap();
    let reason_join = builder
        .add_join(
            left_join,
            reason,
            JoinSpec::new(
                JoinType::Inner,
                vec![JoinKey::new(
                    ColumnRef::input(0, "sr_reason_sk", "sr_reason_sk"),
                    ColumnRef::input(1, "r_reason_sk", "r_reason_sk"),
                )],
            )
            .with_output_projection(vec![
                JoinOutputColumn::left(ColumnRef::input(0, "ss_customer_sk", "ss_customer_sk"), "ss_customer_sk"),
                JoinOutputColumn::left(
                    ColumnRef::input(0, "sr_return_quantity", "sr_return_quantity"),
                    "sr_return_quantity",
                ),
                JoinOutputColumn::left(ColumnRef::input(0, "ss_quantity", "ss_quantity"), "ss_quantity"),
                JoinOutputColumn::left(ColumnRef::input(0, "ss_sales_price", "ss_sales_price"), "ss_sales_price"),
            ])
            .with_build_side(JoinSide::Right),
            reason_join_schema,
        )
        .unwrap();
    let act_sales = Expr::case(
        vec![CaseWhen::new(
            Expr::is_not_null(Expr::column("sr_return_quantity")),
            Expr::binary(Expr::column("ss_quantity"), BinaryOp::Subtract, Expr::column("sr_return_quantity")),
        )],
        Expr::column("ss_quantity"),
    );
    let projected = builder
        .add_projection(
            reason_join,
            vec![
                ProjectionOutput::new(
                    Expr::column("ss_customer_sk"),
                    Field::new("ss_customer_sk", DataType::Int32, false),
                ),
                ProjectionOutput::new(act_sales, Field::new("act_sales", DataType::Int32, true)),
            ],
        )
        .unwrap();
    let aggregate = builder
        .add_aggregate(
            projected,
            AggregateSpec::new(
                vec![Expr::column("ss_customer_sk")],
                vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("act_sales"), "sum_act_sales")],
            ),
            aggregate_schema,
        )
        .unwrap();
    let final_projection = builder
        .add_projection(
            aggregate,
            projection_outputs(
                vec![Expr::column("ss_customer_sk"), Expr::column("sum_act_sales").alias("sumsales")],
                projection_schema,
            ),
        )
        .unwrap();
    let sorted = builder
        .add_sort(
            final_projection,
            SortSpec::new(vec![
                SortExpr::new(Expr::column("sumsales"), false, true),
                SortExpr::new(Expr::column("ss_customer_sk"), false, true),
            ]),
        )
        .unwrap();
    let limited = builder.add_limit(sorted, LimitSpec::fetch(100)).unwrap();
    builder.finish_with_host_sink(limited).unwrap()
}

fn union_aggregate_over_left_join_right_key_plan() -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let returns = builder.add_source(gpu_source("returns", &["return_item", "return_order", "returns_value"]));
    let sales = builder.add_source(gpu_source("sales", &["sales_item", "sales_order", "site_sk"]));
    let direct = builder.add_source(gpu_source("direct", &["site_sk", "returns_value"]));
    let site_dim = builder.add_source(gpu_source("site_dim", &["site_sk"]));

    let left_join_schema = Arc::new(Schema::new(vec![
        Field::new("site_sk", DataType::Int32, true),
        Field::new("returns_value", DataType::Int32, false),
    ]));
    let left_join = builder
        .add_join(
            returns,
            sales,
            JoinSpec::new(
                JoinType::Left,
                vec![
                    JoinKey::new(
                        ColumnRef::input(0, "return_item", "return_item"),
                        ColumnRef::input(1, "sales_item", "sales_item"),
                    ),
                    JoinKey::new(
                        ColumnRef::input(0, "return_order", "return_order"),
                        ColumnRef::input(1, "sales_order", "sales_order"),
                    ),
                ],
            )
            .with_output_projection(vec![
                JoinOutputColumn::right(ColumnRef::input(1, "site_sk", "site_sk"), "site_sk"),
                JoinOutputColumn::left(ColumnRef::input(0, "returns_value", "returns_value"), "returns_value"),
            ])
            .with_build_side(JoinSide::Right),
            Arc::clone(&left_join_schema),
        )
        .unwrap();
    let left_join_projected = builder
        .add_projection(
            left_join,
            vec![
                ProjectionOutput::new(Expr::column("site_sk"), Field::new("site_sk", DataType::Int32, true)),
                ProjectionOutput::new(
                    Expr::column("returns_value"),
                    Field::new("returns_value", DataType::Int32, false),
                ),
            ],
        )
        .unwrap();
    let direct_projected = builder
        .add_projection(
            direct,
            vec![
                ProjectionOutput::new(Expr::column("site_sk"), Field::new("site_sk", DataType::Int32, false)),
                ProjectionOutput::new(
                    Expr::column("returns_value"),
                    Field::new("returns_value", DataType::Int32, false),
                ),
            ],
        )
        .unwrap();
    let union_schema = Arc::new(Schema::new(vec![
        Field::new("site_sk", DataType::Int32, true),
        Field::new("returns_value", DataType::Int32, false),
    ]));
    let union = builder.add_union(vec![left_join_projected, direct_projected], Arc::clone(&union_schema)).unwrap();
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("site_sk", DataType::Int32, true),
        Field::new("sum_returns", DataType::Int32, true),
    ]));
    let aggregate = builder
        .add_aggregate(
            union,
            AggregateSpec::new(
                vec![Expr::column("site_sk")],
                vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("returns_value"), "sum_returns")],
            ),
            Arc::clone(&aggregate_schema),
        )
        .unwrap();
    let root = builder
        .add_join(
            aggregate,
            site_dim,
            JoinSpec::new(
                JoinType::Inner,
                vec![JoinKey::new(
                    ColumnRef::input(0, "site_sk", "site_sk"),
                    ColumnRef::input(1, "site_sk", "site_sk"),
                )],
            )
            .with_output_projection(left_projection(&["site_sk", "sum_returns"])),
            aggregate_schema,
        )
        .unwrap();
    builder.finish_gpu_resident(root).unwrap()
}

#[derive(Debug, Clone, Copy)]
enum MarkerNullFilterPlanVariant {
    Eligible,
    NullableMarker,
    NullsMatch,
    MarkerEscapes,
}

fn marker_null_filter_left_join_plan(variant: MarkerNullFilterPlanVariant) -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(gpu_source("left", &["l_key", "payload"]));
    let right_schema = Arc::new(Schema::new(vec![
        Field::new("r_key", DataType::Int32, matches!(variant, MarkerNullFilterPlanVariant::NullableMarker)),
        Field::new("r_payload", DataType::Int32, false),
    ]));
    let right = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("right", right_schema)));
    let marker_schema = Arc::new(Schema::new(vec![
        Field::new("l_key", DataType::Int32, false),
        Field::new("payload", DataType::Int32, false),
        Field::new("r_key", DataType::Int32, true),
    ]));
    let mut spec = JoinSpec::new(
        JoinType::Left,
        vec![JoinKey::new(ColumnRef::input(0, "l_key", "l_key"), ColumnRef::input(1, "r_key", "r_key"))],
    )
    .with_output_projection(vec![
        JoinOutputColumn::left(ColumnRef::input(0, "l_key", "l_key"), "l_key"),
        JoinOutputColumn::left(ColumnRef::input(0, "payload", "payload"), "payload"),
        JoinOutputColumn::right(ColumnRef::input(1, "r_key", "r_key"), "r_key"),
    ])
    .with_build_side(JoinSide::Right);
    if matches!(variant, MarkerNullFilterPlanVariant::NullsMatch) {
        spec = spec.with_null_equality(JoinNullEquality::NullsMatch);
    }
    let left_join = builder.add_join(left, right, spec, Arc::clone(&marker_schema)).unwrap();
    let filtered = builder.add_filter(left_join, Expr::is_null(Expr::column("r_key"))).unwrap();
    let projected = if matches!(variant, MarkerNullFilterPlanVariant::MarkerEscapes) {
        builder
            .add_projection(
                filtered,
                vec![
                    ProjectionOutput::new(Expr::column("l_key"), Field::new("l_key", DataType::Int32, false)),
                    ProjectionOutput::new(Expr::column("payload"), Field::new("payload", DataType::Int32, false)),
                    ProjectionOutput::new(Expr::column("r_key"), Field::new("r_key", DataType::Int32, true)),
                ],
            )
            .unwrap()
    } else {
        builder
            .add_projection(
                filtered,
                vec![
                    ProjectionOutput::new(Expr::column("l_key"), Field::new("l_key", DataType::Int32, false)),
                    ProjectionOutput::new(Expr::column("payload"), Field::new("payload", DataType::Int32, false)),
                ],
            )
            .unwrap()
    };
    builder.finish_gpu_resident(projected).unwrap()
}

fn outer_join_null_rejection_only_options() -> NativeOptimizerOptions {
    NativeOptimizerOptions {
        outer_join_null_rejection: OuterJoinNullRejectionOptions::default(),
        ..NativeOptimizerOptions::disabled()
    }
}
