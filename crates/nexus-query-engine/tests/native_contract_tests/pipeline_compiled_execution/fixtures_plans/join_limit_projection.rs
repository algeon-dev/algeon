/// Join → Limit → Projection over throwaway paths.
///
// The retired projection/global-limit recognizer no longer decides this
// shape: the composed pipeline admits it, and the compiled `Limit` task
// contract takes one complete frame.
pub(super) fn join_limit_projection_plan() -> QueryPlan {
    let left_schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int32, false),
        Field::new("d", DataType::Int32, false),
        Field::new("v", DataType::Int32, true),
    ]));
    let right_schema = Arc::new(Schema::new(vec![Field::new("d", DataType::Int32, false)]));
    let join_schema =
        Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false), Field::new("v", DataType::Int32, true)]));
    let join_projection_schema =
        Arc::new(Schema::new(vec![Field::new("v", DataType::Int32, true), Field::new("k", DataType::Int32, false)]));

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/projection-global-limit-left.parquet",
        Arc::clone(&left_schema),
        vec!["k".into(), "d".into(), "v".into()],
    )));
    let right = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/projection-global-limit-right.parquet",
        right_schema,
        vec!["d".into()],
    )));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(ColumnRef::input(0, "d", "d"), ColumnRef::input(1, "d", "d"))],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(ColumnRef::input(0, "k", "k"), "k"),
            JoinOutputColumn::left(ColumnRef::input(0, "v", "v"), "v"),
        ])
        .with_build_side(JoinSide::Right),
        Arc::clone(&join_schema),
    );
    let limit = builder.add_limit(join, LimitSpec::fetch(10), Arc::clone(&join_schema));
    let projection = builder.add_projection(
        limit,
        projection_outputs(vec![Expr::column("v"), Expr::column("k")], Arc::clone(&join_projection_schema)),
    );
    builder.finish_with_host_sink(projection)
}

#[test]
fn composed_pipeline_admits_projection_over_join_limit() {
    let plan = join_limit_projection_plan();
    assert_eq!(plan.nodes().iter().filter(|node| matches!(node.kind, PlanNode::Limit(_))).count(), 1);
    let admission =
        bounded_pipeline_admission(&plan).unwrap().expect("single-join limit projection should be admitted");
    assert_eq!(admission.envelope(), nexus_query_engine::capability::CapabilityEnvelope::ComposedPipeline);
    // The retired single-join post-root envelope no longer decides this shape;
    // the compiled `Limit` task takes one complete resident frame.
    let limit_task = admission
        .pipeline()
        .tasks()
        .iter()
        .find(|task| task.kind() == nexus_query_engine::pipeline::PipelineTaskKind::Limit)
        .expect("one limit task");
    assert_eq!(limit_task.accepted_input_form(), nexus_query_engine::pipeline::AcceptedInputForm::Frame);
}
