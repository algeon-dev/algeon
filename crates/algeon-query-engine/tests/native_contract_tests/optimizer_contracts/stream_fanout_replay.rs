#[test]
fn native_optimizer_replays_shared_projected_probe_input_for_join_fanout() {
    let plan = shared_projected_probe_fanout_plan();
    let original_node_count = plan.node_count();
    let result = optimize_query_plan(plan, &NativeOptimizerOptions::default(), rapids_analysis_context(), local_dimension_pruning_resolver());

    assert_eq!(result.change, NativeOptimizerChange::Changed);
    assert_eq!(result.diagnostics.stream_fanout_replay.considered_count, 1);
    assert_eq!(result.diagnostics.stream_fanout_replay.applied_count, 1);
    let candidate = &result.diagnostics.stream_fanout_replay.candidates[0];
    assert!(candidate.applied);
    assert_eq!(candidate.consumer_count, 2);
    assert_eq!(candidate.rejected_reason, None);
    assert!(result.plan.node_count() > original_node_count);
    assert_eq!(max_direct_join_probe_consumer_count(&result.plan), 1);
}

#[test]
fn disabled_native_optimizer_keeps_stream_fanout_replay_off() {
    let plan = shared_projected_probe_fanout_plan();
    let result = optimize_query_plan(plan.clone(), &NativeOptimizerOptions::disabled(), rapids_analysis_context(), local_dimension_pruning_resolver());

    assert_eq!(result.change, NativeOptimizerChange::Unchanged);
    assert_eq!(result.plan, plan);
    assert!(result.diagnostics.stream_fanout_replay.is_empty());
}

fn shared_projected_probe_fanout_plan() -> QueryPlan {
    let mut builder = QueryPlanBuilder::new();
    let fact = builder.add_source(gpu_source("fact", &["key", "qty"]));
    let prefix_dim = builder.add_source(gpu_source("prefix_dim", &["key"]));
    let prefix = builder.add_join(
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
    );
    let projected = builder.add_projection(
        prefix,
        vec![
            ProjectionOutput::new(Expr::column("key"), Field::new("key", DataType::Int32, false)),
            ProjectionOutput::new(Expr::column("qty"), Field::new("qty", DataType::Int32, false)),
        ],
    );
    let first_dim = builder.add_source(gpu_source("first_dim", &["key"]));
    let second_dim = builder.add_source(gpu_source("second_dim", &["key"]));
    let first = builder.add_join(
        projected,
        first_dim,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "key", "key"), ColumnRef::input(1, "key", "key"))],
        )
        .with_output_projection(left_projection(&["key", "qty"]))
        .with_build_side(JoinSide::Right),
        schema(&["key", "qty"]),
    );
    let second = builder.add_join(
        projected,
        second_dim,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "key", "key"), ColumnRef::input(1, "key", "key"))],
        )
        .with_output_projection(left_projection(&["key", "qty"]))
        .with_build_side(JoinSide::Right),
        schema(&["key", "qty"]),
    );
    let union = builder.add_union(vec![first, second], schema(&["key", "qty"]));
    builder.finish_gpu_resident(union)
}
