use super::*;

#[test]
fn native_optimizer_replays_shared_projected_probe_input_for_join_fanout() {
    let plan = shared_projected_probe_fanout_plan();
    let original_node_count = plan.node_count();
    let result = optimize_query_plan(
        plan,
        &NativeOptimizerOptions::default(),
        rapids_analysis_context(PlanningPolicy::default()),
        local_dimension_pruning_resolver(),
    )
    .unwrap();

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    assert_eq!(result.diagnostics.stream_fanout_replay.considered_count, 1);
    assert_eq!(result.diagnostics.stream_fanout_replay.applied_count, 1);
    let candidate = &result.diagnostics.stream_fanout_replay.candidates[0];
    assert!(candidate.applied);
    assert_eq!(candidate.consumer_count, 2);
    assert!(result.plan.node_count() > original_node_count);
    assert_eq!(max_direct_join_probe_consumer_count(&result.plan), 1);
}

#[test]
fn disabled_native_optimizer_keeps_stream_fanout_replay_off() {
    let plan = shared_projected_probe_fanout_plan();
    let result = optimize_query_plan(
        plan.clone(),
        &NativeOptimizerOptions::disabled(),
        rapids_analysis_context(PlanningPolicy::default()),
        local_dimension_pruning_resolver(),
    )
    .unwrap();

    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    assert_eq!(result.plan, plan);
    assert!(result.diagnostics.stream_fanout_replay.is_empty());
}

#[test]
fn native_optimizer_replays_shared_probe_with_residual_consumer_from_budget_share_or_row_limit() {
    let device_budget_bytes = 64 << 30;
    let share = MemoryControlPolicy::retained_state_share_bytes(device_budget_bytes);
    let row_limit = rapids_backend().max_column_elements();
    let rows = 100_000_000;
    // Every byte estimate stays above the small-build byte bound, so
    // build-side selection keeps the anti join building its left side.
    let small_bytes = 1 << 30;
    let lineitem = |bytes, rows| estimated_local_source_with_fields("lineitem", &["key", "supp"], bytes, rows);
    for (case, lineitem, budget, expected_reason, expected_row_evidence) in [
        (
            "below the budget share",
            lineitem(share - 1, rows),
            Some(device_budget_bytes),
            Some(StreamFanoutReplayRejectedReason::ResidualConsumerBelowLimits),
            format!(
                "estimated_rows={rows},estimated_bytes={},budget_share_bytes={share},column_row_limit={row_limit},selection_basis=none,applied=false",
                share - 1
            ),
        ),
        (
            "at the budget share",
            lineitem(share, rows),
            Some(device_budget_bytes),
            None,
            format!(
                "estimated_rows={rows},estimated_bytes={share},budget_share_bytes={share},column_row_limit={row_limit},selection_basis=budget_share_bytes,applied=true"
            ),
        ),
        (
            "at the column row limit",
            lineitem(small_bytes, row_limit),
            Some(device_budget_bytes),
            None,
            format!(
                "estimated_rows={row_limit},estimated_bytes={small_bytes},budget_share_bytes={share},column_row_limit={row_limit},selection_basis=column_row_limit,applied=true"
            ),
        ),
        (
            "unestimated",
            gpu_source("lineitem", &["key", "supp"]),
            Some(device_budget_bytes),
            Some(StreamFanoutReplayRejectedReason::ResidualConsumerUnestimated),
            format!(
                "estimated_rows=none,estimated_bytes=none,budget_share_bytes={share},column_row_limit={row_limit},selection_basis=none,applied=false"
            ),
        ),
        (
            "budget unknown",
            lineitem(share, rows),
            None,
            Some(StreamFanoutReplayRejectedReason::ResidualConsumerBudgetUnknown),
            format!(
                "estimated_rows={rows},estimated_bytes={share},budget_share_bytes=none,column_row_limit={row_limit},selection_basis=none,applied=false"
            ),
        ),
    ] {
        let options = NativeOptimizerOptions {
            stream_fanout_replay: StreamFanoutReplayOptions { device_budget_bytes: budget, ..Default::default() },
            ..Default::default()
        };
        let result = optimize_query_plan(
            shared_probe_with_residual_consumer_plan(lineitem),
            &options,
            rapids_analysis_context(PlanningPolicy::default()),
            local_dimension_pruning_resolver(),
        )
        .unwrap();

        let replay = &result.diagnostics.stream_fanout_replay;
        assert_eq!(replay.considered_count, 1, "{case}");
        let candidate = &replay.candidates[0];
        assert_eq!(candidate.rejected_reason, expected_reason, "{case}");
        assert_eq!(candidate.applied, expected_reason.is_none(), "{case}");
        let expected_probe_consumers = if candidate.applied { 1 } else { 2 };
        assert_eq!(max_direct_join_probe_consumer_count(&result.plan), expected_probe_consumers, "{case}");
        let rows = result.diagnostics.stable_rule_rows();
        let replay_row = rows
            .iter()
            .find(|row| row.starts_with(&format!("rule={STREAM_FANOUT_REPLAY_RULE_NAME},")))
            .expect("stream fanout replay should report its candidate");
        assert!(replay_row.contains(&expected_row_evidence), "{case}: {replay_row}");
    }
}

/// TPC-H Q21's shape: one filtered scan probed by an inner join and by an
/// anti join that builds its left side and has a residual.
fn shared_probe_with_residual_consumer_plan(lineitem: SourceNode) -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let lineitem = builder.add_source(lineitem);
    let filtered = builder
        .add_filter(lineitem, Expr::binary(Expr::column("supp"), BinaryOp::Gt, Expr::literal(LiteralValue::Int32(0))))
        .unwrap();
    let shared = builder
        .add_projection(
            filtered,
            vec![
                ProjectionOutput::new(Expr::column("key"), Field::new("key", DataType::Int32, false)),
                ProjectionOutput::new(Expr::column("supp"), Field::new("supp", DataType::Int32, false)),
            ],
        )
        .unwrap();
    let supplier = builder.add_source(estimated_local_source("supplier", "supp", 1024, 100));
    let supplier_lines = builder
        .add_join(
            shared,
            supplier,
            JoinSpec::new(
                JoinType::Inner,
                vec![JoinKey::new(ColumnRef::input(0, "supp", "supp"), ColumnRef::input(1, "supp", "supp"))],
            )
            .with_output_projection(left_projection(&["key", "supp"]))
            .with_build_side(JoinSide::Right),
            schema(&["key", "supp"]),
        )
        .unwrap();
    let without_other_supplier = builder
        .add_join(
            supplier_lines,
            shared,
            JoinSpec::new(
                JoinType::LEFT_ANTI,
                vec![JoinKey::new(ColumnRef::input(0, "key", "key"), ColumnRef::input(1, "key", "key"))],
            )
            .with_residual_predicate(Expr::binary(
                Expr::input_column(0, "supp", "supp"),
                BinaryOp::NotEq,
                Expr::input_column(1, "supp", "supp"),
            ))
            .with_output_projection(left_projection(&["key", "supp"]))
            .with_build_side(JoinSide::Left),
            schema(&["key", "supp"]),
        )
        .unwrap();
    builder.finish_gpu_resident(without_other_supplier).unwrap()
}

fn shared_projected_probe_fanout_plan() -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let fact = builder.add_source(gpu_source("fact", &["key", "qty"]));
    let prefix_dim = builder.add_source(gpu_source("prefix_dim", &["key"]));
    let prefix = builder
        .add_join(
            fact,
            prefix_dim,
            JoinSpec::new(
                JoinType::Inner,
                vec![JoinKey::new(ColumnRef::input(0, "key", "key"), ColumnRef::input(1, "key", "key"))],
            )
            .with_output_projection(vec![
                JoinOutputColumn::left(ColumnRef::input(0, "key", "key"), "key"),
                JoinOutputColumn::left(ColumnRef::input(0, "qty", "qty"), "qty"),
            ])
            .with_build_side(JoinSide::Right),
            schema(&["key", "qty"]),
        )
        .unwrap();
    let projected = builder
        .add_projection(
            prefix,
            vec![
                ProjectionOutput::new(Expr::column("key"), Field::new("key", DataType::Int32, false)),
                ProjectionOutput::new(Expr::column("qty"), Field::new("qty", DataType::Int32, false)),
            ],
        )
        .unwrap();
    let first_dim = builder.add_source(gpu_source("first_dim", &["key"]));
    let second_dim = builder.add_source(gpu_source("second_dim", &["key"]));
    let first = builder
        .add_join(
            projected,
            first_dim,
            JoinSpec::new(
                JoinType::Inner,
                vec![JoinKey::new(ColumnRef::input(0, "key", "key"), ColumnRef::input(1, "key", "key"))],
            )
            .with_output_projection(left_projection(&["key", "qty"]))
            .with_build_side(JoinSide::Right),
            schema(&["key", "qty"]),
        )
        .unwrap();
    let second = builder
        .add_join(
            projected,
            second_dim,
            JoinSpec::new(
                JoinType::Inner,
                vec![JoinKey::new(ColumnRef::input(0, "key", "key"), ColumnRef::input(1, "key", "key"))],
            )
            .with_output_projection(left_projection(&["key", "qty"]))
            .with_build_side(JoinSide::Right),
            schema(&["key", "qty"]),
        )
        .unwrap();
    let union = builder.add_union(vec![first, second], schema(&["key", "qty"])).unwrap();
    builder.finish_gpu_resident(union).unwrap()
}
