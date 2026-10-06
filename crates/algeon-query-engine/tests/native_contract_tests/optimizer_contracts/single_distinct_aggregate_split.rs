#[test]
fn single_distinct_aggregate_split_runs_in_the_optimizer_and_its_plan_is_admitted() {
    let mut builder = QueryPlanBuilder::new();
    let source =
        builder.add_source(estimated_local_source_with_fields("events", &["user_id", "day", "amount"], 1_024, 10));
    let aggregate = builder
        .add_aggregate(
            source,
            AggregateSpec::new(
                vec![Expr::column("user_id")],
                vec![
                    AggregateExpr::new(AggregateFunction::NUnique, Expr::column("day"), "active_days"),
                    AggregateExpr::new(AggregateFunction::Sum, Expr::column("amount"), "total"),
                    AggregateExpr::new(AggregateFunction::Count, Expr::column("amount"), "events"),
                    AggregateExpr::new(AggregateFunction::Avg, Expr::column("amount"), "mean"),
                ],
            ),
            Arc::new(Schema::new(vec![
                Field::new("user_id", DataType::Int32, false),
                Field::new("active_days", DataType::Int64, false),
                Field::new("total", DataType::Int64, true),
                Field::new("events", DataType::Int64, false),
                Field::new("mean", DataType::Float64, true),
            ])),
        )
        .unwrap();
    let plan = builder.finish_with_host_sink(aggregate).unwrap();

    let result = optimize_query_plan(
        plan.clone(),
        &NativeOptimizerOptions::default(),
        rapids_analysis_context(PlanningPolicy::default()),
        local_dimension_pruning_resolver(),
    )
    .unwrap();

    let rows = result.diagnostics.stable_rule_rows();
    assert!(
        rows.contains(
            &"rule=single_distinct_aggregate_split,considered_count=1,applied_count=1,candidates=rule=single_distinct_aggregate_split,aggregate_node_id=n1,distinct_column=day,applied=true,rejected_reason=none"
                .to_string()
        ),
        "{rows:?}"
    );
    assert_eq!(result.plan.output_schema(), plan.output_schema());
    admit_query_plan(&result.plan, StatisticalAggregateExecutionMode::default(), PlanningPolicy::default())
        .expect("both aggregates and the AVG projection pass validation, capability analysis and compilation");
}
