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
fn native_optimizer_flips_left_semi_keyed_residual_join_to_small_left_build_side() {
    let plan = estimated_semi_anti_residual_join_plan(
        JoinType::LeftSemi,
        JoinSide::Right,
        16 * 1024 * 1024,
        100_000,
        256 * 1024 * 1024,
        10_000_000,
    );

    let result = optimize_query_plan(plan, &NativeOptimizerOptions::default());

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    assert_eq!(result.diagnostics.join_build_side_selection.considered_count, 1);
    assert_eq!(result.diagnostics.join_build_side_selection.applied_count, 1);
    let candidate = &result.diagnostics.join_build_side_selection.candidates[0];
    assert_eq!(candidate.before_build_side, JoinSide::Right);
    assert_eq!(candidate.after_build_side, Some(JoinSide::Left));
    assert!(candidate.applied);

    let PlanNode::Join(join) = &result.plan.root_node().kind else {
        panic!("optimized root should remain the left-semi residual join");
    };
    assert_eq!(join.spec.join_type, JoinType::LeftSemi);
    assert!(join.spec.residual_predicate.is_some());
    assert!(join.spec.output_projection.iter().all(|column| column.side == JoinSide::Left));
    assert_eq!(join.spec.build_side, JoinSide::Left);
}

#[test]
fn native_optimizer_flips_left_anti_keyed_residual_join_to_small_left_build_side() {
    let plan = estimated_semi_anti_residual_join_plan(
        JoinType::LeftAnti,
        JoinSide::Right,
        16 * 1024 * 1024,
        100_000,
        256 * 1024 * 1024,
        10_000_000,
    );

    let result = optimize_query_plan(plan, &NativeOptimizerOptions::default());

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    assert_eq!(result.diagnostics.join_build_side_selection.considered_count, 1);
    assert_eq!(result.diagnostics.join_build_side_selection.applied_count, 1);
    let candidate = &result.diagnostics.join_build_side_selection.candidates[0];
    assert_eq!(candidate.before_build_side, JoinSide::Right);
    assert_eq!(candidate.after_build_side, Some(JoinSide::Left));
    assert!(candidate.applied);

    let PlanNode::Join(join) = &result.plan.root_node().kind else {
        panic!("optimized root should remain the left-anti residual join");
    };
    assert_eq!(join.spec.join_type, JoinType::LeftAnti);
    assert!(join.spec.residual_predicate.is_some());
    assert!(join.spec.output_projection.iter().all(|column| column.side == JoinSide::Left));
    assert_eq!(join.spec.build_side, JoinSide::Left);
}

#[test]
fn join_build_side_selection_keeps_residual_semi_anti_join_when_sizes_do_not_improve_it() {
    for join_type in [JoinType::LeftSemi, JoinType::LeftAnti] {
        let plan = estimated_semi_anti_residual_join_plan(
            join_type,
            JoinSide::Right,
            256 * 1024 * 1024,
            10_000_000,
            256 * 1024 * 1024,
            10_000_000,
        );

        let result = optimize_query_plan(plan, &NativeOptimizerOptions::default());

        assert_eq!(result.change, NativeOptimizerChange::Unchanged, "{join_type}");
        let candidate = result
            .diagnostics
            .join_build_side_selection
            .candidates
            .iter()
            .find(|candidate| candidate.before_build_side == JoinSide::Right)
            .expect("residual semi/anti join should be considered");
        assert_eq!(candidate.after_build_side, None);
        assert!(!candidate.applied);
        assert_eq!(candidate.rejected_reason, Some(JoinBuildSideSelectionRejectedReason::NoBetterBuildSide));
        let PlanNode::Join(join) = &result.plan.root_node().kind else {
            panic!("optimized root should remain the residual semi/anti join");
        };
        assert_eq!(join.spec.build_side, JoinSide::Right);
    }
}

#[test]
fn disabled_native_optimizer_keeps_residual_semi_anti_join_build_side_selection_off() {
    for join_type in [JoinType::LeftSemi, JoinType::LeftAnti] {
        let plan = estimated_semi_anti_residual_join_plan(
            join_type,
            JoinSide::Right,
            16 * 1024 * 1024,
            100_000,
            256 * 1024 * 1024,
            10_000_000,
        );

        let result = optimize_query_plan(plan.clone(), &NativeOptimizerOptions::disabled());

        assert_eq!(result.plan, plan, "{join_type}");
        assert_eq!(result.change, NativeOptimizerChange::Unchanged, "{join_type}");
        assert!(result.diagnostics.join_build_side_selection.is_empty(), "{join_type}");
    }
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
fn join_build_side_selection_uses_advisory_join_output_rows_when_conservative_size_is_unknown() {
    let advised = advisory_rows(2_000);
    let plan = advised_join_over_fact_plan(Some(advised), 5 * 1024 * 1024 * 1024, 100_000_000, JoinSide::Right);

    let result = optimize_query_plan(plan, &NativeOptimizerOptions::default());

    let candidate = join_fed_build_side_candidate(&result.diagnostics.join_build_side_selection);
    assert!(candidate.applied);
    assert_eq!(candidate.after_build_side, Some(JoinSide::Left));
    assert_eq!(candidate.selection_basis, Some(JoinBuildSideSelectionBasis::AdvisoryRows));
    assert_eq!(candidate.left_estimated_rows, None);
    assert_eq!(candidate.left_estimated_bytes, None);
    assert_eq!(candidate.left_advisory_rows, Some(2_000));
    assert_eq!(candidate.left_advisory_bytes, None);
    assert_eq!(candidate.right_estimated_rows, Some(100_000_000));
    assert_eq!(candidate.right_advisory_rows, None);
    assert_eq!(candidate.rejected_reason, None);
    assert_eq!(root_join_build_side(&result.plan), JoinSide::Left);

    let plan = advised_join_over_fact_plan(None, 5 * 1024 * 1024 * 1024, 100_000_000, JoinSide::Right);

    let result = optimize_query_plan(plan, &NativeOptimizerOptions::default());

    let candidate = join_fed_build_side_candidate(&result.diagnostics.join_build_side_selection);
    assert!(!candidate.applied);
    assert_eq!(candidate.after_build_side, None);
    assert_eq!(candidate.selection_basis, None);
    assert_eq!(candidate.left_advisory_rows, None);
    assert_eq!(candidate.rejected_reason, Some(JoinBuildSideSelectionRejectedReason::NoBetterBuildSide));
    assert_eq!(root_join_build_side(&result.plan), JoinSide::Right);
}

#[test]
fn join_build_side_selection_keeps_current_build_side_when_advisory_output_is_larger_than_known_side() {
    let plan =
        advised_join_over_fact_plan(Some(advisory_rows(500_000_000)), 5 * 1024 * 1024 * 1024, 100_000_000, JoinSide::Right);

    let result = optimize_query_plan(plan, &NativeOptimizerOptions::default());

    let candidate = join_fed_build_side_candidate(&result.diagnostics.join_build_side_selection);
    assert!(!candidate.applied);
    assert_eq!(candidate.after_build_side, None);
    assert_eq!(candidate.selection_basis, Some(JoinBuildSideSelectionBasis::AdvisoryRows));
    assert_eq!(candidate.left_advisory_rows, Some(500_000_000));
    assert_eq!(candidate.rejected_reason, Some(JoinBuildSideSelectionRejectedReason::NoBetterBuildSide));
    assert_eq!(root_join_build_side(&result.plan), JoinSide::Right);
}

#[test]
fn join_build_side_selection_prefers_conservative_evidence_over_advisory_estimates() {
    // A conservatively small current build side is kept even when the other
    // side advertises a tinier advisory output.
    let plan = advised_join_over_fact_plan(Some(advisory_rows(10)), 32 * 1024 * 1024, 500_000, JoinSide::Right);

    let result = optimize_query_plan(plan, &NativeOptimizerOptions::default());

    let candidate = join_fed_build_side_candidate(&result.diagnostics.join_build_side_selection);
    assert!(!candidate.applied);
    assert_eq!(candidate.selection_basis, Some(JoinBuildSideSelectionBasis::SmallCurrentSide));
    assert_eq!(candidate.left_advisory_rows, Some(10));
    assert_eq!(root_join_build_side(&result.plan), JoinSide::Right);

    // A side that carries both a conservative bound and a much smaller
    // advisory estimate is compared by its bound: the advisory never
    // shrinks known evidence.
    let mut builder = QueryPlanBuilder::new();
    let semi_left = builder.add_source(estimated_local_source("semi_left", "s_key", 1024 * 1024 * 1024, 50_000_000));
    let semi_right = builder.add_source(estimated_local_source("semi_right", "r_key", 1024, 10));
    let semi = builder.add_join(
        semi_left,
        semi_right,
        JoinSpec::new(
            JoinType::LeftSemi,
            vec![JoinKey::new(ColumnRef::input(0, "s_key", "s_key"), ColumnRef::input(1, "r_key", "r_key"))],
        )
        .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "s_key", "s_key"), "s_key")])
        .with_build_side(JoinSide::Right)
        .with_advisory_output_statistics(AdvisoryOutputStatistics {
            rows: AdvisoryStatistic::Inexact(100),
            bytes: AdvisoryStatistic::Inexact(1_024),
            provenance: AdvisoryStatisticsProvenance::FrontendPlannerEstimate,
        }),
        schema(&["s_key"]),
    );
    let fact = builder.add_source(estimated_local_source_with_fields(
        "fact",
        &["f_key", "f_val"],
        512 * 1024 * 1024,
        20_000_000,
    ));
    let outer = builder.add_join(
        semi,
        fact,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "s_key", "s_key"), ColumnRef::input(1, "f_key", "f_key"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "s_key", "s_key"), "s_key"),
            JoinOutputColumn::right(ColumnRef::input(1, "f_val", "f_val"), "f_val"),
        ])
        .with_build_side(JoinSide::Left),
        schema(&["s_key", "f_val"]),
    );
    let plan = builder.finish_gpu_resident(outer);

    let result = optimize_query_plan(plan, &NativeOptimizerOptions::default());

    let candidate = result
        .diagnostics
        .join_build_side_selection
        .candidates
        .iter()
        .find(|candidate| candidate.left_advisory_bytes.is_some())
        .expect("semi-fed outer join should be a build-side candidate");
    assert!(candidate.applied);
    assert_eq!(candidate.after_build_side, Some(JoinSide::Right));
    assert_eq!(candidate.selection_basis, Some(JoinBuildSideSelectionBasis::ConservativeBytes));
    assert_eq!(candidate.left_estimated_bytes, Some(1024 * 1024 * 1024));
    assert_eq!(candidate.left_advisory_bytes, Some(1_024));
    assert_eq!(root_join_build_side(&result.plan), JoinSide::Right);
}

#[test]
fn join_build_side_selection_keeps_small_current_build_side_over_grouped_aggregate_keyset() {
    let mut builder = QueryPlanBuilder::new();
    let fact = builder.add_source(estimated_local_source_with_fields(
        "sales_for_grouped_date_keyset",
        &["sold_date_sk", "sales_price"],
        512 * 1024 * 1024,
        10_000_000,
    ));
    let grouped_fact = builder.add_aggregate(
        fact,
        AggregateSpec::new(vec![Expr::column("sold_date_sk").alias("date_sk")], Vec::new()),
        schema(&["date_sk"]),
    );
    let date = builder.add_source(estimated_local_source("small_date_dim", "d_date_sk", 1024 * 1024, 365));
    let join = builder.add_join(
        grouped_fact,
        date,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "date_sk", "date_sk"),
                ColumnRef::input(1, "d_date_sk", "d_date_sk"),
            )],
        )
        .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "date_sk", "date_sk"), "date_sk")])
        .with_build_side(JoinSide::Right),
        schema(&["date_sk"]),
    );
    let plan = builder.finish_gpu_resident(join);
    let options = NativeOptimizerOptions {
        join_build_side_selection: JoinBuildSideSelectionOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(plan, &options);

    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    let candidate = result
        .diagnostics
        .join_build_side_selection
        .candidates
        .iter()
        .find(|candidate| candidate.candidate_node_id == join)
        .expect("date join candidate should be considered");
    assert!(!candidate.applied);
    assert_eq!(candidate.before_build_side, JoinSide::Right);
    assert_eq!(candidate.after_build_side, None);
    assert_eq!(candidate.left_estimated_rows, None);
    assert_eq!(candidate.left_estimated_bytes, None);
    assert_eq!(candidate.right_estimated_rows, Some(365));
    assert_eq!(candidate.right_estimated_bytes, Some(1024 * 1024));

    let PlanNode::Join(join) = &result.plan.root_node().kind else {
        panic!("optimized root should remain the date join");
    };
    assert_eq!(join.spec.build_side, JoinSide::Right);
}

#[test]
fn join_build_side_selection_prefers_small_other_side_over_grouped_aggregate_keyset() {
    let mut builder = QueryPlanBuilder::new();
    let fact = builder.add_source(estimated_local_source_with_fields(
        "sales_for_grouped_week_keyset",
        &["sold_date_sk", "sales_price"],
        512 * 1024 * 1024,
        10_000_000,
    ));
    let grouped_fact = builder.add_aggregate(
        fact,
        AggregateSpec::new(vec![Expr::column("sold_date_sk").alias("date_sk")], Vec::new()),
        schema(&["date_sk"]),
    );
    let week_keyset = builder.add_source(estimated_local_source("tiny_week_keyset", "d_date_sk", 1024, 7));
    let join = builder.add_join(
        grouped_fact,
        week_keyset,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "date_sk", "date_sk"),
                ColumnRef::input(1, "d_date_sk", "d_date_sk"),
            )],
        )
        .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "date_sk", "date_sk"), "date_sk")])
        .with_build_side(JoinSide::Left),
        schema(&["date_sk"]),
    );
    let plan = builder.finish_gpu_resident(join);
    let options = NativeOptimizerOptions {
        join_build_side_selection: JoinBuildSideSelectionOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(plan, &options);

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    let candidate = result
        .diagnostics
        .join_build_side_selection
        .candidates
        .iter()
        .find(|candidate| candidate.candidate_node_id == join)
        .expect("date keyset join candidate should be considered");
    assert!(candidate.applied);
    assert_eq!(candidate.before_build_side, JoinSide::Left);
    assert_eq!(candidate.after_build_side, Some(JoinSide::Right));
    assert_eq!(candidate.left_estimated_rows, None);
    assert_eq!(candidate.left_estimated_bytes, None);
    assert_eq!(candidate.right_estimated_rows, Some(7));
    assert_eq!(candidate.right_estimated_bytes, Some(1024));

    let PlanNode::Join(join) = &result.plan.root_node().kind else {
        panic!("optimized root should remain the date keyset join");
    };
    assert_eq!(join.spec.build_side, JoinSide::Right);
}

#[test]
fn join_build_side_selection_prefers_bounded_left_semi_other_side_over_grouped_aggregate_keyset() {
    let mut builder = QueryPlanBuilder::new();
    let fact = builder.add_source(estimated_local_source_with_fields(
        "sales_for_left_semi_week_keyset",
        &["sold_date_sk", "sales_price"],
        512 * 1024 * 1024,
        10_000_000,
    ));
    let grouped_fact = builder.add_aggregate(
        fact,
        AggregateSpec::new(vec![Expr::column("sold_date_sk").alias("date_sk")], Vec::new()),
        schema(&["date_sk"]),
    );
    let dates = builder.add_source(estimated_local_source_with_fields(
        "date_dim_for_week_keyset",
        &["d_date_sk", "d_week_seq"],
        1024 * 1024,
        365,
    ));
    let week = builder.add_source(estimated_local_source("week_seq_keyset", "week_seq", 1024, 1));
    let week_dates = builder.add_join(
        dates,
        week,
        JoinSpec::new(
            JoinType::LeftSemi,
            vec![JoinKey::new(
                ColumnRef::input(0, "d_week_seq", "d_week_seq"),
                ColumnRef::input(1, "week_seq", "week_seq"),
            )],
        )
        .with_output_projection(vec![JoinOutputColumn::left(
            ColumnRef::input(0, "d_date_sk", "d_date_sk"),
            "d_date_sk",
        )])
        .with_build_side(JoinSide::Right),
        schema(&["d_date_sk"]),
    );
    let join = builder.add_join(
        grouped_fact,
        week_dates,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "date_sk", "date_sk"),
                ColumnRef::input(1, "d_date_sk", "d_date_sk"),
            )],
        )
        .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "date_sk", "date_sk"), "date_sk")])
        .with_build_side(JoinSide::Left),
        schema(&["date_sk"]),
    );
    let plan = builder.finish_gpu_resident(join);
    let options = NativeOptimizerOptions {
        join_build_side_selection: JoinBuildSideSelectionOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(plan, &options);

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    let candidate = result
        .diagnostics
        .join_build_side_selection
        .candidates
        .iter()
        .find(|candidate| candidate.candidate_node_id == join)
        .expect("left-semi keyset join candidate should be considered");
    assert!(candidate.applied);
    assert_eq!(candidate.before_build_side, JoinSide::Left);
    assert_eq!(candidate.after_build_side, Some(JoinSide::Right));
    assert_eq!(candidate.left_estimated_rows, None);
    assert_eq!(candidate.left_estimated_bytes, None);
    assert_eq!(candidate.right_estimated_rows, Some(365));
    assert_eq!(candidate.right_estimated_bytes, Some(1024 * 1024));

    let PlanNode::Join(join) = &result.plan.root_node().kind else {
        panic!("optimized root should remain the grouped fact/date keyset join");
    };
    assert_eq!(join.spec.build_side, JoinSide::Right);
}

#[test]
fn join_build_side_selection_prefers_bounded_left_anti_other_side_over_grouped_aggregate_keyset() {
    let mut builder = QueryPlanBuilder::new();
    let fact = builder.add_source(estimated_local_source_with_fields(
        "sales_for_left_anti_week_keyset",
        &["sold_date_sk", "sales_price"],
        512 * 1024 * 1024,
        10_000_000,
    ));
    let grouped_fact = builder.add_aggregate(
        fact,
        AggregateSpec::new(vec![Expr::column("sold_date_sk").alias("date_sk")], Vec::new()),
        schema(&["date_sk"]),
    );
    let dates = builder.add_source(estimated_local_source_with_fields(
        "date_dim_for_anti_week_keyset",
        &["d_date_sk", "d_week_seq"],
        1024 * 1024,
        365,
    ));
    let excluded_week = builder.add_source(estimated_local_source("excluded_week_seq_keyset", "week_seq", 1024, 1));
    let remaining_dates = builder.add_join(
        dates,
        excluded_week,
        JoinSpec::new(
            JoinType::LeftAnti,
            vec![JoinKey::new(
                ColumnRef::input(0, "d_week_seq", "d_week_seq"),
                ColumnRef::input(1, "week_seq", "week_seq"),
            )],
        )
        .with_output_projection(vec![JoinOutputColumn::left(
            ColumnRef::input(0, "d_date_sk", "d_date_sk"),
            "d_date_sk",
        )])
        .with_build_side(JoinSide::Right),
        schema(&["d_date_sk"]),
    );
    let join = builder.add_join(
        grouped_fact,
        remaining_dates,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "date_sk", "date_sk"),
                ColumnRef::input(1, "d_date_sk", "d_date_sk"),
            )],
        )
        .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "date_sk", "date_sk"), "date_sk")])
        .with_build_side(JoinSide::Left),
        schema(&["date_sk"]),
    );
    let plan = builder.finish_gpu_resident(join);
    let options = NativeOptimizerOptions {
        join_build_side_selection: JoinBuildSideSelectionOptions::default(),
        ..NativeOptimizerOptions::disabled()
    };

    let result = optimize_query_plan(plan, &options);

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    let candidate = result
        .diagnostics
        .join_build_side_selection
        .candidates
        .iter()
        .find(|candidate| candidate.candidate_node_id == join)
        .expect("left-anti keyset join candidate should be considered");
    assert!(candidate.applied);
    assert_eq!(candidate.before_build_side, JoinSide::Left);
    assert_eq!(candidate.after_build_side, Some(JoinSide::Right));
    assert_eq!(candidate.left_estimated_rows, None);
    assert_eq!(candidate.left_estimated_bytes, None);
    assert_eq!(candidate.right_estimated_rows, Some(365));
    assert_eq!(candidate.right_estimated_bytes, Some(1024 * 1024));

    let PlanNode::Join(join) = &result.plan.root_node().kind else {
        panic!("optimized root should remain the grouped fact/date keyset join");
    };
    assert_eq!(join.spec.build_side, JoinSide::Right);
}

#[test]
fn native_optimizer_revisits_build_side_after_aggregate_join_pushdown_creates_grouped_keyset() {
    let mut builder = QueryPlanBuilder::new();
    let sales = builder.add_source(estimated_local_source_with_fields(
        "sales",
        &["item_sk", "sales_price"],
        512 * 1024 * 1024,
        10_000_000,
    ));
    let item = builder.add_source(estimated_local_source_with_fields(
        "item",
        &["i_item_sk", "i_class"],
        256 * 1024 * 1024,
        2_000_000,
    ));
    let join = builder.add_join(
        sales,
        item,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "item_sk", "item_sk"),
                ColumnRef::input(1, "i_item_sk", "i_item_sk"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::right(ColumnRef::input(1, "i_class", "i_class"), "i_class"),
            JoinOutputColumn::left(ColumnRef::input(0, "sales_price", "sales_price"), "sales_price"),
        ])
        .with_build_side(JoinSide::Right),
        schema(&["i_class", "sales_price"]),
    );
    let aggregate = builder.add_aggregate(
        join,
        AggregateSpec::new(
            vec![Expr::column("i_class")],
            vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("sales_price"), "class_sales")],
        ),
        schema(&["i_class", "class_sales"]),
    );
    let plan = builder.finish_with_host_sink(aggregate);

    let result = optimize_query_plan(plan, &NativeOptimizerOptions::default());

    assert_eq!(result.diagnostics.aggregate_join_pushdown.applied_count, 1);
    assert!(
        result.diagnostics.join_build_side_selection.candidates.iter().any(|candidate| {
            candidate.applied
                && candidate.before_build_side == JoinSide::Right
                && candidate.after_build_side == Some(JoinSide::Left)
        }),
        "post-aggregate build-side selection should flip the reduced join to the grouped keyset side"
    );

    let PlanNode::Aggregate(final_aggregate) = &result.plan.root_node().kind else {
        panic!("optimized root should remain the final aggregate");
    };
    let PlanNode::Join(reduced_join) =
        &result.plan.node(final_aggregate.input).expect("final aggregate input should exist").kind
    else {
        panic!("final aggregate should consume the reduced join");
    };
    assert_eq!(reduced_join.spec.build_side, JoinSide::Left);
    let PlanNode::Aggregate(preaggregate) =
        &result.plan.node(reduced_join.inputs[0]).expect("left reduced-join input should exist").kind
    else {
        panic!("left reduced-join input should be the pushed aggregate");
    };
    assert!(preaggregate.spec.group_exprs.iter().any(|expr| matches!(
        expr,
        Expr::Alias { alias, .. } if alias == "item_sk"
    )));
}

#[test]
fn native_optimizer_revisits_build_side_after_dimension_pruning_changes_source_size() {
    let dir = unique_contract_temp_dir("dimension-pruning-build-side");
    let fact_path = dir.join("fact.parquet");
    let dim_path = dir.join("dim.parquet");
    let fact_schema = Arc::new(Schema::new(vec![Field::new("fact_key", DataType::Int32, false)]));
    let dim_schema = Arc::new(Schema::new(vec![
        Field::new("dim_key", DataType::Int32, false),
        Field::new("d_year", DataType::Int32, false),
    ]));
    write_i32_record_batch_parquet(
        &fact_path,
        Arc::clone(&fact_schema),
        vec![Int32Array::from(vec![1, 100, 200, 300])],
        1,
    );
    write_i32_record_batch_parquet(
        &dim_path,
        Arc::clone(&dim_schema),
        vec![Int32Array::from(vec![100]), Int32Array::from(vec![2000])],
        1,
    );

    let mut builder = QueryPlanBuilder::new();
    let fact = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
        vec![
            LocalParquetFile::new(&fact_path)
                .with_projected_row_group_bytes(vec![80 * 1024 * 1024; 4], vec![80 * 1024 * 1024; 4])
                .with_row_group_rows(vec![1; 4]),
        ],
        Arc::clone(&fact_schema),
        vec!["fact_key".to_string()],
    )));
    let dim = builder.add_source(SourceNode::LocalParquet(
        LocalParquetSource::from_files(
            vec![
                LocalParquetFile::new(&dim_path)
                    .with_projected_row_group_bytes(vec![96 * 1024 * 1024], vec![96 * 1024 * 1024])
                    .with_row_group_rows(vec![1]),
            ],
            Arc::clone(&dim_schema),
            vec!["dim_key".to_string(), "d_year".to_string()],
        )
        .with_reader_predicate(Expr::binary(
            Expr::column("d_year"),
            BinaryOp::Eq,
            Expr::literal(LiteralValue::Int32(2000)),
        )),
    ));
    let join = builder.add_join(
        fact,
        dim,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "fact_key", "fact_key"), ColumnRef::input(1, "dim_key", "dim_key"))],
        )
        .with_output_projection(vec![JoinOutputColumn::left(ColumnRef::input(0, "fact_key", "fact_key"), "fact_key")])
        .with_build_side(JoinSide::Right),
        Arc::clone(&fact_schema),
    );
    let plan = builder.finish_gpu_resident(join);

    let result = optimize_query_plan(plan, &NativeOptimizerOptions::default());

    assert_eq!(result.diagnostics.dimension_derived_row_group_pruning.applied_count, 1);
    let first_build_side_candidate = result
        .diagnostics
        .join_build_side_selection
        .candidates
        .iter()
        .find(|candidate| candidate.candidate_node_id == join && !candidate.applied)
        .expect("initial build-side pass should keep the smaller unpruned dimension side");
    assert_eq!(first_build_side_candidate.left_estimated_bytes, Some(320 * 1024 * 1024));
    assert_eq!(first_build_side_candidate.right_estimated_bytes, Some(96 * 1024 * 1024));

    let post_pruning_candidate = result
        .diagnostics
        .join_build_side_selection
        .candidates
        .iter()
        .find(|candidate| candidate.candidate_node_id == join && candidate.applied)
        .expect("post-pruning build-side pass should reconsider the pruned fact side");
    assert_eq!(post_pruning_candidate.before_build_side, JoinSide::Right);
    assert_eq!(post_pruning_candidate.after_build_side, Some(JoinSide::Left));
    assert_eq!(post_pruning_candidate.left_estimated_bytes, Some(80 * 1024 * 1024));
    assert_eq!(post_pruning_candidate.right_estimated_bytes, Some(96 * 1024 * 1024));

    let PlanNode::Join(join) = &result.plan.root_node().kind else {
        panic!("optimized root should remain the fact/dimension join");
    };
    assert_eq!(join.spec.build_side, JoinSide::Left);

    let PlanNode::Source(source) = &result.plan.node(fact).expect("fact source should remain").kind else {
        panic!("fact node should remain a source");
    };
    let SourceNode::LocalParquet(source) = source.as_ref() else {
        panic!("fact node should remain local parquet");
    };
    assert_eq!(source.files[0].row_groups.as_deref(), Some(&[1][..]));

    fs::remove_dir_all(dir).expect("temporary build-side pruning fixture should be removable");
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

fn advisory_rows(rows: u64) -> AdvisoryOutputStatistics {
    AdvisoryOutputStatistics {
        rows: AdvisoryStatistic::Inexact(rows),
        bytes: AdvisoryStatistic::Absent,
        provenance: AdvisoryStatisticsProvenance::FrontendPlannerEstimate,
    }
}

/// Builds `projection(dim ⋈ mid) ⋈ fact` where only the frontend advisory
/// describes the inner join output; both sources of the inner join keep
/// conservative estimates, but the join output itself has none.
fn advised_join_over_fact_plan(
    advisory: Option<AdvisoryOutputStatistics>,
    fact_bytes: u64,
    fact_rows: u64,
    build_side: JoinSide,
) -> nexus_query_engine::plan::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let dim = builder.add_source(estimated_local_source("dim", "d_key", 128 * 1024 * 1024, 15_000_000));
    let mid = builder.add_source(estimated_local_source_with_fields(
        "mid",
        &["m_key", "m_id"],
        1024 * 1024 * 1024,
        70_000_000,
    ));
    let mut inner_spec = JoinSpec::new(
        JoinType::Inner,
        vec![JoinKey::new(ColumnRef::input(0, "d_key", "d_key"), ColumnRef::input(1, "m_key", "m_key"))],
    )
    .with_output_projection(vec![JoinOutputColumn::right(ColumnRef::input(1, "m_id", "m_id"), "m_id")])
    .with_build_side(JoinSide::Left);
    if let Some(advisory) = advisory {
        inner_spec = inner_spec.with_advisory_output_statistics(advisory);
    }
    let inner = builder.add_join(dim, mid, inner_spec, schema(&["m_id"]));
    let projected = builder.add_projection(
        inner,
        vec![ProjectionOutput::new(Expr::column("m_id"), Field::new("m_id", DataType::Int32, false))],
    );
    let fact = builder.add_source(estimated_local_source_with_fields("fact", &["f_mid", "f_val"], fact_bytes, fact_rows));
    let outer = builder.add_join(
        projected,
        fact,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "m_id", "m_id"), ColumnRef::input(1, "f_mid", "f_mid"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "m_id", "m_id"), "m_id"),
            JoinOutputColumn::right(ColumnRef::input(1, "f_val", "f_val"), "f_val"),
        ])
        .with_build_side(build_side),
        schema(&["m_id", "f_val"]),
    );
    builder.finish_gpu_resident(outer)
}

/// Finds the candidate for the outer join fed by another join. Node ids are
/// renumbered by earlier passes, so the candidate is identified by the only
/// evidence shape it can have: no conservative estimate on its left input.
fn join_fed_build_side_candidate(
    diagnostics: &JoinBuildSideSelectionDiagnostics,
) -> &nexus_query_engine::optimizer::JoinBuildSideSelectionCandidateDiagnostics {
    let mut candidates = diagnostics.candidates.iter().filter(|candidate| candidate.left_estimated_rows.is_none());
    let candidate = candidates.next().expect("join-fed outer join should be a build-side candidate");
    assert!(candidates.next().is_none(), "exactly one join-fed candidate expected: {diagnostics:?}");
    candidate
}

fn root_join_build_side(plan: &nexus_query_engine::plan::QueryPlan) -> JoinSide {
    let PlanNode::Join(join) = &plan.root_node().kind else {
        panic!("optimized root should remain the outer join");
    };
    join.spec.build_side
}

fn estimated_inner_equi_join_plan(left_bytes: u64, right_bytes: u64) -> nexus_query_engine::plan::QueryPlan {
    estimated_inner_join_plan(JoinSide::Right, left_bytes, 1, right_bytes, 1)
}

fn estimated_inner_join_plan(
    build_side: JoinSide,
    left_bytes: u64,
    left_rows: u64,
    right_bytes: u64,
    right_rows: u64,
) -> nexus_query_engine::plan::QueryPlan {
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
) -> nexus_query_engine::plan::QueryPlan {
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

fn estimated_semi_anti_residual_join_plan(
    join_type: JoinType,
    build_side: JoinSide,
    left_bytes: u64,
    left_rows: u64,
    right_bytes: u64,
    right_rows: u64,
) -> nexus_query_engine::plan::QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(estimated_local_source_with_fields(
        "left_semi_anti_residual",
        &["left_key", "left_payload"],
        left_bytes,
        left_rows,
    ));
    let right = builder.add_source(estimated_local_source_with_fields(
        "right_semi_anti_residual",
        &["right_key", "right_payload"],
        right_bytes,
        right_rows,
    ));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(
            join_type,
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
            JoinOutputColumn::left(ColumnRef::input(0, "left_payload", "left_payload"), "left_payload"),
        ])
        .with_build_side(build_side),
        schema(&["left_key", "left_payload"]),
    );
    builder.finish_gpu_resident(join)
}
