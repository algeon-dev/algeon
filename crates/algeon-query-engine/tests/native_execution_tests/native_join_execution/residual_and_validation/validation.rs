use std::sync::Arc;

use super::*;

#[test]
fn native_query_rejects_invalid_cross_join_before_source_read() {
    let left_schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    let right_schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    let output_schema = Arc::new(Schema::new(vec![
        Field::new("left_id", DataType::Int32, false),
        Field::new("right_id", DataType::Int32, true),
    ]));
    let left_source = LocalParquetSource::from_path(
        "/tmp/does-not-need-to-exist-for-cross-join-left.parquet",
        Arc::clone(&left_schema),
        vec!["id".into()],
    );
    let right_source = LocalParquetSource::from_path(
        "/tmp/does-not-need-to-exist-for-cross-join-right.parquet",
        Arc::clone(&right_schema),
        vec!["id".into()],
    );

    let mut builder = QueryPlanBuilder::new();
    let left = builder.add_source(SourceNode::LocalParquet(left_source));
    let right = builder.add_source(SourceNode::LocalParquet(right_source));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(
            JoinType::Cross,
            vec![JoinKey::new(
                query_engine::expr::ColumnRef::input(0, "id", "left_id"),
                query_engine::expr::ColumnRef::input(1, "id", "right_id"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(query_engine::expr::ColumnRef::input(0, "id", "left_id"), "left_id"),
            JoinOutputColumn::right(query_engine::expr::ColumnRef::input(1, "id", "right_id"), "right_id"),
        ]),
        output_schema,
    );
    let plan = builder.finish_with_host_sink(join);
    let context = backend_context();

    let error = match execute_query_plan(&context, &plan) {
        Ok(_) => panic!("native validation should reject cross joins with equality keys before source read"),
        Err(error) => error,
    };
    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.operation(), "query_engine.join.validate");
    assert_eq!(error.fact_value("key_count"), Some("1"));
    assert!(error.fact_value("path").is_none());
}

#[test]
fn native_query_rejects_cross_join_with_residual_before_source_read() {
    let error = residual_validation_error(
        JoinType::Cross,
        vec![],
        Expr::binary(Expr::input_column(0, "id", "left_id"), BinaryOp::Eq, Expr::input_column(1, "id", "right_id")),
    );

    assert_eq!(error.code(), QueryErrorCode::UnsupportedPlanShape);
    assert_eq!(error.operation(), "query_engine.join.validate");
    assert_eq!(error.fact_value("join_type"), Some("cross"));
    assert!(error.fact_value("path").is_none());
}

#[test]
fn native_query_rejects_duplicate_join_output_names_before_runtime_materialization() {
    let left_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("n_name", DataType::Utf8, false),
    ]));
    let right_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("n_name", DataType::Utf8, false),
    ]));
    let duplicate_join_schema = Arc::new(Schema::new(vec![
        Field::new("n_name", DataType::Utf8, false),
        Field::new("n_name", DataType::Utf8, false),
    ]));
    let projected_schema = Arc::new(Schema::new(vec![
        Field::new("supp_nation", DataType::Utf8, false),
        Field::new("cust_nation", DataType::Utf8, false),
    ]));

    let mut builder = QueryPlanBuilder::new();
    let left =
        builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("left_input", Arc::clone(&left_schema))));
    let right =
        builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("right_input", Arc::clone(&right_schema))));
    let join = builder.add_join(
        left,
        right,
        JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                query_engine::expr::ColumnRef::input(0, "id", "left_id"),
                query_engine::expr::ColumnRef::input(1, "id", "right_id"),
            )],
        )
        .with_output_projection(vec![
            JoinOutputColumn::left(query_engine::expr::ColumnRef::input(0, "n_name", "supp_n_name"), "n_name"),
            JoinOutputColumn::right(query_engine::expr::ColumnRef::input(1, "n_name", "cust_n_name"), "n_name"),
        ])
        .with_null_equality(JoinNullEquality::NullsDoNotMatch)
        .with_build_side(JoinSide::Right),
        duplicate_join_schema,
    );
    let projection = builder.add_projection(
        join,
        projection_outputs(
            vec![Expr::column("n_name").alias("supp_nation"), Expr::column("n_name").alias("cust_nation")],
            projected_schema,
        ),
    );
    let plan = builder.finish_with_host_sink(projection);
    let context = backend_context();

    let error = match execute_query_plan(&context, &plan) {
        Ok(_) => panic!("native validation should reject duplicate join output names before runtime materialization"),
        Err(error) => error,
    };
    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.operation(), "query_engine.plan.validate_output_names");
    assert_eq!(error.fact_value("duplicate_output_name"), Some("n_name"));
    assert_eq!(error.fact_value("first_output_index"), Some("0"));
    assert_eq!(error.fact_value("duplicate_output_index"), Some("1"));
    assert_eq!(error.fact_value("output_names"), Some("n_name,n_name"));
    assert_eq!(error.fact_value("node_kind"), Some("join"));
    assert_eq!(error.fact_value("schema_behavior"), Some("join"));
    assert!(error.fact_value("input_name").is_none());
}
