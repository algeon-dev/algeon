use std::sync::Arc;

use query_planning::expr::{CaseWhen, DatePart};

use super::*;

#[test]
fn native_query_rejects_unqualified_residual_column_before_source_read() {
    let error = residual_validation_error(
        JoinType::Inner,
        vec![id_join_key()],
        Expr::binary(Expr::column("id"), BinaryOp::Eq, Expr::input_column(1, "id", "right_id")),
    );

    // An unqualified residual column names no join input: a malformed plan,
    // while the one-sided predicate below is a legal backend gap.
    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.operation(), "query_engine.join.validate_residual");
    assert_eq!(error.fact_value("column"), Some("id"));
    assert_eq!(error.fact_value("qualifier"), Some("none"));
    assert!(error.fact_value("path").is_none());
}

#[test]
fn native_query_rejects_one_sided_residual_predicate_before_source_read() {
    let error = residual_validation_error(
        JoinType::Inner,
        vec![id_join_key()],
        Expr::binary(Expr::input_column(0, "id", "left_id"), BinaryOp::Gt, Expr::literal(LiteralValue::Int32(1))),
    );

    assert_eq!(error.code(), QueryErrorCode::UnsupportedExpression);
    assert_eq!(error.operation(), "query_engine.join.validate_residual");
    assert_eq!(error.fact_value("references_left"), Some("true"));
    assert_eq!(error.fact_value("references_right"), Some("false"));
    assert!(error.fact_value("path").is_none());
}

#[test]
fn native_query_rejects_non_boolean_residual_predicate_before_source_read() {
    let error = residual_validation_error(
        JoinType::Inner,
        vec![id_join_key()],
        Expr::binary(Expr::input_column(0, "id", "left_id"), BinaryOp::Add, Expr::input_column(1, "id", "right_id")),
    );

    assert_eq!(error.code(), QueryErrorCode::UnsupportedType);
    assert_eq!(error.operation(), "query_engine.plan.validate_expr");
    assert_eq!(error.fact_value("context"), Some("join residual predicate"));
    assert!(error.fact_value("path").is_none());
}

#[test]
fn native_query_rejects_join_residual_in_list_with_non_literal_values_before_source_read() {
    let error = residual_validation_error(
        JoinType::Inner,
        vec![id_join_key()],
        Expr::binary(
            Expr::in_list(Expr::input_column(0, "id", "left_id"), vec![Expr::input_column(1, "id", "right_id")]),
            BinaryOp::And,
            Expr::binary(Expr::input_column(0, "id", "left_id"), BinaryOp::Eq, Expr::input_column(1, "id", "right_id")),
        ),
    );

    assert_eq!(error.code(), QueryErrorCode::UnsupportedExpression);
    assert_eq!(error.operation(), "query_engine.join.validate_residual");
    assert!(error.message().contains("IN list values must be literals"), "unexpected error: {error}");
    assert!(error.fact_value("path").is_none());
}

#[test]
fn native_query_rejects_unsupported_join_residual_cast_before_source_read() {
    let error = residual_validation_error(
        JoinType::Inner,
        vec![id_join_key()],
        Expr::binary(
            Expr::cast(Expr::input_column(0, "id", "left_id"), DataType::Int32),
            BinaryOp::Eq,
            Expr::input_column(1, "id", "right_id"),
        ),
    );

    assert_eq!(error.code(), QueryErrorCode::UnsupportedExpression);
    assert_eq!(error.operation(), "query_engine.join.validate_residual");
    assert_eq!(error.fact_value("target_type"), Some("Int32"));
    assert!(error.fact_value("path").is_none());
}

#[test]
fn native_query_rejects_unsupported_residual_expression_families_before_source_read() {
    let cases = vec![
        ("LIKE", Expr::like(Expr::input_column(0, "id", "left_id"), "1%")),
        ("SUBSTRING", Expr::substring(Expr::input_column(0, "id", "left_id"), 1, Some(1))),
        ("EXTRACT", Expr::extract(DatePart::Year, Expr::input_column(0, "id", "left_id"))),
    ];

    for (family, residual) in cases {
        let error = residual_validation_error(JoinType::Inner, vec![id_join_key()], residual);
        assert_eq!(error.code(), QueryErrorCode::UnsupportedExpression, "{family}");
        assert_eq!(error.operation(), "query_engine.join.validate_residual", "{family}");
        assert!(
            error.fact_value("expression").is_some_and(|expression| expression.contains(family)),
            "{family} should be visible in expression fact: {:?}",
            error.fact_value("expression")
        );
        assert!(error.fact_value("path").is_none(), "{family}");
    }
}

#[test]
fn native_query_rejects_post_join_date_trunc_residual_before_source_read() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("event_ts", DataType::Timestamp(TimeUnit::Nanosecond, None), false),
    ]));
    let left_ts = Expr::date_trunc(
        DatePart::Day,
        Expr::input_column(0, "event_ts", "left_event_ts"),
        DataType::Timestamp(TimeUnit::Nanosecond, None),
    );
    let right_ts = Expr::date_trunc(
        DatePart::Day,
        Expr::input_column(1, "event_ts", "right_event_ts"),
        DataType::Timestamp(TimeUnit::Nanosecond, None),
    );
    let residual = Expr::binary(
        Expr::case(
            vec![CaseWhen::new(
                Expr::binary(
                    Expr::input_column(0, "id", "left_id"),
                    BinaryOp::Eq,
                    Expr::input_column(1, "id", "right_id"),
                ),
                left_ts,
            )],
            right_ts.clone(),
        ),
        BinaryOp::Eq,
        right_ts,
    );

    let error = residual_validation_error_with_schema(
        JoinType::Inner,
        vec![id_join_key()],
        residual,
        schema,
        vec!["id", "event_ts"],
    );

    assert_eq!(error.code(), QueryErrorCode::UnsupportedExpression);
    assert_eq!(error.operation(), "query_engine.join.validate_residual");
    assert!(
        error.fact_value("expression").is_some_and(|expression| expression.contains("DATE_TRUNC")),
        "DATE_TRUNC should be visible in expression fact: {:?}",
        error.fact_value("expression")
    );
    assert!(error.fact_value("path").is_none());
}

#[test]
fn native_query_rejects_post_join_extract_new_fields_before_source_read() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("event_ts", DataType::Timestamp(TimeUnit::Nanosecond, None), false),
    ]));
    let residual = Expr::binary(
        Expr::case(
            vec![CaseWhen::new(
                Expr::binary(
                    Expr::input_column(0, "id", "left_id"),
                    BinaryOp::Eq,
                    Expr::input_column(1, "id", "right_id"),
                ),
                Expr::extract(DatePart::Hour, Expr::input_column(0, "event_ts", "left_event_ts")),
            )],
            Expr::extract(DatePart::Hour, Expr::input_column(1, "event_ts", "right_event_ts")),
        ),
        BinaryOp::Gt,
        Expr::literal(LiteralValue::Int32(0)),
    );

    let error = residual_validation_error_with_schema(
        JoinType::Inner,
        vec![id_join_key()],
        residual,
        schema,
        vec!["id", "event_ts"],
    );

    assert_eq!(error.code(), QueryErrorCode::UnsupportedExpression);
    assert_eq!(error.operation(), "query_engine.join.validate_residual");
    assert_eq!(error.fact_value("field"), Some("HOUR"));
    assert!(error.fact_value("path").is_none());
}
