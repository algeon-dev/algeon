#[test]
fn outer_join_null_rejection_converts_left_join_when_inner_join_key_rejects_right_nulls() {
    let plan = inner_join_over_projected_left_join_right_key_plan();
    let options = NativeOptimizerOptions {
        join_filter_reassociation: nexus_query_engine::JoinFilterReassociationOptions::disabled(),
        ..NativeOptimizerOptions::default()
    };

    let result = optimize_query_plan(plan, &options);

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
fn outer_join_null_rejection_keeps_left_join_when_key_comes_from_preserved_side() {
    let plan = inner_join_over_projected_left_join_left_key_plan();

    let result = optimize_query_plan(plan.clone(), &outer_join_null_rejection_only_options());

    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    assert_eq!(result.plan, plan);
    assert!(result.diagnostics.outer_join_null_rejection.is_empty());
}

#[test]
fn outer_join_null_rejection_rejects_shared_outer_join_output() {
    let plan = shared_left_join_output_with_one_null_rejecting_consumer_plan();

    let result = optimize_query_plan(plan.clone(), &outer_join_null_rejection_only_options());

    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    assert_eq!(result.plan, plan);
    assert_eq!(result.diagnostics.outer_join_null_rejection.considered_count, 1);
    assert_eq!(result.diagnostics.outer_join_null_rejection.applied_count, 0);
    assert_eq!(
        result.diagnostics.outer_join_null_rejection.candidates[0].rejected_reason,
        Some(OuterJoinNullRejectionRejectedReason::SharedOuterJoinOutput)
    );
}

#[test]
fn outer_join_null_rejection_converts_marker_null_filter_to_left_anti() {
    let plan = marker_null_filter_left_join_plan(MarkerNullFilterPlanVariant::Eligible);

    let result = optimize_query_plan(plan, &outer_join_null_rejection_only_options());

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    assert_eq!(result.diagnostics.outer_join_null_rejection.considered_count, 1);
    assert_eq!(result.diagnostics.outer_join_null_rejection.applied_count, 1);

    let PlanNode::Projection(projection) = &result.plan.root_node().kind else {
        panic!("root should remain the final projection");
    };
    let PlanNode::Join(join) = &result.plan.node(projection.input).expect("anti join").kind else {
        panic!("projection should consume rewritten anti join");
    };
    assert_eq!(join.spec.join_type, JoinType::LeftAnti);
    assert_eq!(join.spec.build_side, JoinSide::Right);
    assert!(join.spec.output_projection.iter().all(|column| column.side == JoinSide::Left));
    assert_eq!(join.spec.output_projection.len(), 2);
}

#[test]
fn outer_join_null_rejection_converts_marker_null_filter_when_key_schema_is_nullable() {
    let plan = marker_null_filter_left_join_plan(MarkerNullFilterPlanVariant::NullableMarker);

    let result = optimize_query_plan(plan, &outer_join_null_rejection_only_options());

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    let PlanNode::Projection(projection) = &result.plan.root_node().kind else {
        panic!("root should remain the final projection");
    };
    let PlanNode::Join(join) = &result.plan.node(projection.input).expect("anti join").kind else {
        panic!("projection should consume rewritten anti join");
    };
    assert_eq!(join.spec.join_type, JoinType::LeftAnti);
}

#[test]
fn outer_join_null_rejection_keeps_marker_null_filter_with_nulls_match_join() {
    let plan = marker_null_filter_left_join_plan(MarkerNullFilterPlanVariant::NullsMatch);

    let result = optimize_query_plan(plan.clone(), &outer_join_null_rejection_only_options());

    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    assert_eq!(result.plan, plan);
    assert!(result.diagnostics.outer_join_null_rejection.is_empty());
}

#[test]
fn outer_join_null_rejection_keeps_marker_null_filter_when_marker_escapes_projection() {
    let plan = marker_null_filter_left_join_plan(MarkerNullFilterPlanVariant::MarkerEscapes);

    let result = optimize_query_plan(plan.clone(), &outer_join_null_rejection_only_options());

    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    assert_eq!(result.plan, plan);
    assert!(result.diagnostics.outer_join_null_rejection.is_empty());
}

fn inner_join_over_projected_left_join_right_key_plan() -> nexus_query_engine::QueryPlan {
    inner_join_over_projected_left_join_key_plan("r_key")
}

fn inner_join_over_projected_left_join_left_key_plan() -> nexus_query_engine::QueryPlan {
    inner_join_over_projected_left_join_key_plan("l_key")
}

fn inner_join_over_projected_left_join_key_plan(rejecting_key: &str) -> nexus_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(gpu_source("left", &["l_key", "payload"]));
    let right = builder.add_source(gpu_source("right", &["r_key", "r_reason"]));
    let dim = builder.add_source(gpu_source("dim", &[rejecting_key]));
    let left_join_schema = schema(&["l_key", "payload", "r_key", "r_reason"]);
    let left_join = builder.add_join(
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
    );
    let projected = builder.add_projection(
        left_join,
        vec![
            ProjectionOutput::new(Expr::column("l_key"), Field::new("l_key", DataType::Int32, false)),
            ProjectionOutput::new(Expr::column("payload"), Field::new("payload", DataType::Int32, false)),
            ProjectionOutput::new(Expr::column("r_key"), Field::new("r_key", DataType::Int32, true)),
            ProjectionOutput::new(Expr::column("r_reason"), Field::new("r_reason", DataType::Int32, true)),
        ],
    );
    let root_schema = Arc::new(Schema::new(vec![
        Field::new("l_key", DataType::Int32, false),
        Field::new("payload", DataType::Int32, false),
        Field::new("r_key", DataType::Int32, true),
        Field::new("r_reason", DataType::Int32, true),
    ]));
    let root = builder.add_join(
        projected,
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
    );
    builder.finish_gpu_resident(root)
}

fn shared_left_join_output_with_one_null_rejecting_consumer_plan() -> nexus_query_engine::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(gpu_source("left", &["l_key"]));
    let right = builder.add_source(gpu_source("right", &["r_key"]));
    let dim = builder.add_source(gpu_source("dim", &["r_key"]));
    let left_join_schema = schema(&["l_key", "r_key"]);
    let left_join = builder.add_join(
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
    );
    let rejecting = builder.add_join(
        left_join,
        dim,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "r_key", "r_key"), ColumnRef::input(1, "r_key", "r_key"))],
        )
        .with_output_projection(left_projection(&["l_key", "r_key"])),
        Arc::clone(&left_join_schema),
    );
    let root = builder.add_join(
        left_join,
        rejecting,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "l_key", "l_key"), ColumnRef::input(1, "l_key", "l_key"))],
        )
        .with_output_projection(left_projection(&["l_key", "r_key"])),
        left_join_schema,
    );
    builder.finish_gpu_resident(root)
}

#[derive(Debug, Clone, Copy)]
enum MarkerNullFilterPlanVariant {
    Eligible,
    NullableMarker,
    NullsMatch,
    MarkerEscapes,
}

fn marker_null_filter_left_join_plan(variant: MarkerNullFilterPlanVariant) -> nexus_query_engine::QueryPlan {
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
    let left_join = builder.add_join(left, right, spec, Arc::clone(&marker_schema));
    let filtered = builder.add_filter(left_join, Expr::is_null(Expr::column("r_key")), marker_schema);
    let projected = if matches!(variant, MarkerNullFilterPlanVariant::MarkerEscapes) {
        builder.add_projection(
            filtered,
            vec![
                ProjectionOutput::new(Expr::column("l_key"), Field::new("l_key", DataType::Int32, false)),
                ProjectionOutput::new(Expr::column("payload"), Field::new("payload", DataType::Int32, false)),
                ProjectionOutput::new(Expr::column("r_key"), Field::new("r_key", DataType::Int32, true)),
            ],
        )
    } else {
        builder.add_projection(
            filtered,
            vec![
                ProjectionOutput::new(Expr::column("l_key"), Field::new("l_key", DataType::Int32, false)),
                ProjectionOutput::new(Expr::column("payload"), Field::new("payload", DataType::Int32, false)),
            ],
        )
    };
    builder.finish_gpu_resident(projected)
}

fn outer_join_null_rejection_only_options() -> NativeOptimizerOptions {
    NativeOptimizerOptions {
        outer_join_null_rejection: nexus_query_engine::OuterJoinNullRejectionOptions::default(),
        ..NativeOptimizerOptions::disabled()
    }
}
