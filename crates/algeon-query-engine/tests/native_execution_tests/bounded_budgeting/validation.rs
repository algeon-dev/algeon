use std::sync::Arc;

use super::*;

#[test]
fn bounded_native_query_rejects_unsupported_expression_before_source_read() {
    let input_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("value", DataType::Int64, false),
    ]));
    let aggregated_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int32, false),
        Field::new("sum_value", DataType::Int64, true),
    ]));
    let source = LocalParquetSource::from_path(
        "/tmp/does-not-need-to-exist-for-expression-preflight.parquet",
        Arc::clone(&input_schema),
        vec!["key".into(), "value".into()],
    );
    let unsupported_filter =
        Expr::binary(Expr::column("key"), BinaryOp::Eq, Expr::literal(LiteralValue::Null(DataType::Binary)));
    let aggregate = AggregateSpec::new(
        vec![Expr::column("key")],
        vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("value"), "sum_value")],
    );

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let filter = builder.add_filter(source, unsupported_filter).unwrap();
    let aggregate = builder.add_aggregate(filter, aggregate, aggregated_schema).unwrap();
    let plan = builder.finish_with_host_sink(aggregate).unwrap();
    let context = backend_context_with_grant(1024 * 1024);

    let error = match execute_query_plan(&context, &plan) {
        Ok(_) => panic!("bounded execution should reject unsupported expressions before source read"),
        Err(error) => error,
    };

    assert_eq!(error.code(), QueryErrorCode::UnsupportedType);
    assert_eq!(error.operation(), "query_engine.plan.validate_expr");
    assert!(error.fact_value("path").is_none());
}
