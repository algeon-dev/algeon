pub(super) fn adaptive_memory_history_key_probe_plan(
    value_type: DataType,
    aggregate_function: AggregateFunction,
    predicate_op: BinaryOp,
) -> QueryPlan {
    let source_schema = Arc::new(Schema::new(vec![
        Field::new("adaptive_key", DataType::Int32, false),
        Field::new("adaptive_value", value_type.clone(), true),
    ]));
    let aggregate_value_type = match aggregate_function {
        AggregateFunction::Count | AggregateFunction::NUnique => DataType::Int64,
        AggregateFunction::Sum | AggregateFunction::Min | AggregateFunction::Max | AggregateFunction::Avg => {
            value_type.clone()
        },
        AggregateFunction::StddevSamp
        | AggregateFunction::StddevPop
        | AggregateFunction::VarSamp
        | AggregateFunction::VarPop => DataType::Float64,
    };
    let aggregate_schema = Arc::new(Schema::new(vec![
        Field::new("adaptive_key", DataType::Int32, false),
        Field::new("adaptive_metric", aggregate_value_type, true),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::new(
        "/tmp/adaptive-key-secret-literal.parquet",
        source_schema,
        vec!["adaptive_key".into(), "adaptive_value".into()],
    )));
    let filter = builder.add_filter(
        source,
        Expr::binary(Expr::column("adaptive_value"), predicate_op, literal_expr_for_type(&value_type)),
        Arc::new(Schema::new(vec![
            Field::new("adaptive_key", DataType::Int32, false),
            Field::new("adaptive_value", value_type.clone(), true),
        ])),
    );
    let aggregate = builder.add_aggregate(
        filter,
        AggregateSpec::new(
            vec![Expr::column("adaptive_key")],
            vec![AggregateExpr::new(aggregate_function, Expr::column("adaptive_value"), "adaptive_metric")],
        ),
        Arc::clone(&aggregate_schema),
    );
    builder.finish_with_host_sink(aggregate)
}
