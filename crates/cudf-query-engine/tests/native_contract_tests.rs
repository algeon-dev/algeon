use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema};
use cudf_query_engine::{
    AggregateExpr, AggregateFunction, AggregateSpec, BinaryOp, BoundaryMetrics, DataPlacement, Expr, LimitSpec,
    LiteralValue, LocalParquetSource, PlanNode, QueryError, QueryErrorCode, QueryErrorKind, QueryErrorStatus,
    QueryMetrics, QueryPlanBuilder, ResultSink, SchemaBehavior, SortExpr, SortSpec, SourceNode,
};

#[test]
fn native_plan_can_represent_proven_benchmark_shape() {
    let input_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::UInt32, false),
        Field::new("payload", DataType::Int64, false),
        Field::new("m00", DataType::Float64, false),
        Field::new("m01", DataType::Float64, false),
    ]));
    let projected_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::UInt32, false),
        Field::new("x00", DataType::Float64, false),
    ]));
    let aggregated_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::UInt32, false),
        Field::new("sum_x00", DataType::Float64, true),
    ]));

    let source = LocalParquetSource::new(
        "/tmp/phase4.parquet",
        Arc::clone(&input_schema),
        vec!["key".to_string(), "payload".to_string(), "m00".to_string(), "m01".to_string()],
    )
    .with_file_size_bytes(3_235_196_281);
    let filter = Expr::binary(Expr::column("payload"), BinaryOp::GtEq, Expr::literal(LiteralValue::Int64(10)));
    let projection =
        vec![Expr::column("key"), Expr::binary(Expr::column("m00"), BinaryOp::Add, Expr::column("m01")).alias("x00")];
    let aggregate = AggregateSpec::new(
        vec![Expr::column("key")],
        vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("x00"), "sum_x00")],
    );
    let sort = SortSpec::new(vec![
        SortExpr::new(Expr::column("sum_x00"), true, false),
        SortExpr::new(Expr::column("key"), false, true),
    ]);

    let mut builder = QueryPlanBuilder::new();
    let source = builder.add_source(SourceNode::LocalParquet(source));
    let filter = builder.add_filter(source, filter, Arc::clone(&input_schema));
    let projection = builder.add_projection(filter, projection, projected_schema);
    let aggregate = builder.add_aggregate(projection, aggregate, Arc::clone(&aggregated_schema));
    let sort = builder.add_sort(aggregate, sort, Arc::clone(&aggregated_schema));
    let limit = builder.add_limit(sort, LimitSpec::fetch(100), Arc::clone(&aggregated_schema));
    let plan = builder.finish_with_host_sink(limit);

    assert_eq!(plan.node_count(), 6);
    assert_eq!(plan.output_schema().fields().len(), 2);
    assert_eq!(plan.root_id(), limit);
    assert!(matches!(plan.sink(), ResultSink::HostArrow { input, .. } if *input == limit));
    assert!(plan.sink().materializes_host_boundary());
    assert!(plan.nodes().iter().all(|node| node.facts.placement == DataPlacement::Device));
    assert_eq!(plan.node(source).unwrap().facts.schema_behavior, SchemaBehavior::SourceOutput);
    assert_eq!(plan.node(filter).unwrap().facts.schema_behavior, SchemaBehavior::PreservesInput);
    assert_eq!(plan.node(projection).unwrap().facts.schema_behavior, SchemaBehavior::Projection);
    assert_eq!(plan.node(aggregate).unwrap().facts.schema_behavior, SchemaBehavior::Aggregate);
    assert_eq!(plan.node(sort).unwrap().facts.output_ordering.len(), 2);
    assert_eq!(plan.node(limit).unwrap().facts.output_ordering.len(), 2);
    assert!(matches!(plan.root_node().kind, PlanNode::Limit(_)));
    assert!(plan.to_string().contains("LocalParquetSource"));
    assert!(plan.to_string().contains("Limit input=n4 skip=0 fetch=100"));
    assert!(plan.to_string().contains("Sink id=sink input=n5 mode=host_arrow materialization=device_to_host"));
}

#[test]
fn query_error_codes_and_facts_are_stable_without_datafusion() {
    let error = QueryError::new(
        QueryErrorCode::UnsupportedExpression,
        "cudf_query_engine.plan.validate",
        "expression is not supported by native cudf query engine",
    )
    .with_fact("expression", "Utf8 LIKE")
    .with_fact("reason", "unsupported_expression");

    assert_eq!(error.code(), QueryErrorCode::UnsupportedExpression);
    assert_eq!(error.code().as_str(), "unsupported_expression");
    assert_eq!(error.kind(), QueryErrorKind::Unsupported);
    assert_eq!(error.status(), QueryErrorStatus::Permanent);
    assert_eq!(error.fact_value("reason"), Some("unsupported_expression"));
    assert!(error.to_string().contains("code=unsupported_expression"));
    assert!(error.to_string().contains("expression=\"Utf8 LIKE\""));
}

#[test]
fn query_metrics_make_host_device_boundaries_explicit() {
    let mut metrics = QueryMetrics::default();
    metrics.record_parquet_source(Some(3_235_196_281), 12, 34);
    metrics.boundary.record_device_to_host_materialization(100, 8192);

    assert_eq!(metrics.parquet_files, 1);
    assert_eq!(metrics.parquet_bytes_known, 3_235_196_281);
    assert_eq!(metrics.row_groups_selected, 12);
    assert_eq!(metrics.read_columns, 34);
    assert_eq!(metrics.boundary.host_to_device_uploads, 0);
    assert_eq!(metrics.boundary.device_to_host_materializations, 1);
    assert_eq!(metrics.boundary.device_to_host_materialized_rows, 100);

    let mut boundary = BoundaryMetrics::default();
    boundary.record_host_to_device_upload(10, 1024);
    assert_eq!(boundary.host_to_device_uploads, 1);
    assert_eq!(boundary.host_to_device_uploaded_bytes, 1024);
}
