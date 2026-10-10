use query_planning::pipeline::{AcceptedInputForm, PipelineTaskKind};

use super::*;

/// Join → Limit → Projection over throwaway paths.
// The retired projection/global-limit recognizer no longer decides this
// shape: the composed pipeline admits it, and the compiled `Limit` task
// contract takes one complete frame.
fn join_limit_projection_plan() -> QueryPlan {
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
    let join = builder
        .add_join(
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
        )
        .unwrap();
    let limit = builder.add_limit(join, LimitSpec::fetch(10)).unwrap();
    let projection = builder
        .add_projection(
            limit,
            projection_outputs(vec![Expr::column("v"), Expr::column("k")], Arc::clone(&join_projection_schema)),
        )
        .unwrap();
    builder.finish_with_host_sink(projection).unwrap()
}

/// Two joins sharing one filtered dimension scan, under an aggregate and limit.
///
/// This is the TPC-DS q32/q38/q87/q92/q97 shape: a `date_dim`-style scan is
/// filtered once and fanned out to several join branches. A deferrable Parquet
/// read hands the filter a source stream and the filter re-stages it; the
/// joins share the filter's output, so the filter reads the scan once into
/// the frame both joins build from.
pub(in crate::exec::bounded::compiled::execution_tests) fn shared_filtered_dimension_fan_out_plan(
    dimension_path: &Path,
    fact_paths: [&Path; 2],
) -> QueryPlan {
    let fact_schema =
        Arc::new(Schema::new(vec![Field::new("d", DataType::Int32, false), Field::new("v", DataType::Int32, true)]));
    let dim_schema = Arc::new(Schema::new(vec![
        Field::new("d", DataType::Int32, false),
        Field::new("year", DataType::Int32, false),
    ]));
    let joined_schema =
        Arc::new(Schema::new(vec![Field::new("d", DataType::Int32, false), Field::new("v", DataType::Int32, true)]));
    let grouped_schema = Arc::new(Schema::new(vec![
        Field::new("d", DataType::Int32, false),
        Field::new("total", DataType::Int64, true),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let dimension = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        dimension_path,
        Arc::clone(&dim_schema),
        vec!["d".into(), "year".into()],
    )));
    let filtered_dimension = builder
        .add_filter(
            dimension,
            Expr::binary(Expr::column("year"), BinaryOp::Eq, Expr::literal(LiteralValue::Int32(2000))),
        )
        .unwrap();
    let join_against_dimension = |builder: &mut QueryPlanBuilder, path: &Path| {
        let fact = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
            path,
            Arc::clone(&fact_schema),
            vec!["d".into(), "v".into()],
        )));
        builder
            .add_join(
                fact,
                filtered_dimension,
                JoinSpec::new(
                    JoinType::Inner,
                    vec![JoinKey::new(ColumnRef::input(0, "d", "d"), ColumnRef::input(1, "d", "d"))],
                )
                .with_output_projection(vec![
                    JoinOutputColumn::left(ColumnRef::input(0, "d", "d"), "d"),
                    JoinOutputColumn::left(ColumnRef::input(0, "v", "v"), "v"),
                ])
                .with_build_side(JoinSide::Right),
                Arc::clone(&joined_schema),
            )
            .unwrap()
    };
    let store_sales = join_against_dimension(&mut builder, fact_paths[0]);
    let web_sales = join_against_dimension(&mut builder, fact_paths[1]);
    let combined = builder
        .add_join(
            store_sales,
            web_sales,
            JoinSpec::new(
                JoinType::Inner,
                vec![JoinKey::new(ColumnRef::input(0, "d", "d"), ColumnRef::input(1, "d", "d"))],
            )
            .with_output_projection(vec![
                JoinOutputColumn::left(ColumnRef::input(0, "d", "d"), "d"),
                JoinOutputColumn::left(ColumnRef::input(0, "v", "v"), "v"),
            ])
            .with_build_side(JoinSide::Right),
            Arc::clone(&joined_schema),
        )
        .unwrap();
    let grouped = builder
        .add_aggregate(
            combined,
            AggregateSpec::new(
                vec![Expr::column("d")],
                vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("v"), "total")],
            ),
            Arc::clone(&grouped_schema),
        )
        .unwrap();
    let limit = builder.add_limit(grouped, LimitSpec::fetch(100)).unwrap();
    builder.finish_with_host_sink(limit).unwrap()
}

#[test]
fn composed_pipeline_admits_projection_over_join_limit() {
    let plan = join_limit_projection_plan();
    assert_eq!(plan.nodes().iter().filter(|node| matches!(node.kind, PlanNode::Limit(_))).count(), 1);
    let admission =
        bounded_pipeline_admission(&plan).unwrap().expect("single-join limit projection should be admitted");
    // The retired single-join post-root envelope no longer decides this shape;
    // the compiled `Limit` task takes one complete resident frame.
    let limit_task = admission
        .pipeline()
        .tasks()
        .iter()
        .find(|task| task.kind() == PipelineTaskKind::Limit)
        .expect("one limit task");
    assert_eq!(limit_task.accepted_input_form(), AcceptedInputForm::Frame);
}

#[test]
fn composed_pipeline_admits_a_filtered_source_scan_shared_by_several_joins() {
    // A bare shared scan was always admitted; the same scan with a filter above
    // it must be too, because the filter only re-stages the scan's stream.
    let plan = shared_filtered_dimension_fan_out_plan(
        Path::new("/tmp/shared-filtered-dimension.parquet"),
        [Path::new("/tmp/shared-filtered-store-sales.parquet"), Path::new("/tmp/shared-filtered-web-sales.parquet")],
    );
    assert!(
        bounded_pipeline_admission(&plan).unwrap().is_some(),
        "a filtered dimension scan shared by several joins should be admitted"
    );
}
