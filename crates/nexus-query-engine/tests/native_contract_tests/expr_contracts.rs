use super::*;

#[test]
fn native_expr_can_represent_boolean_not() {
    let expr = !Expr::binary(Expr::column("payload"), BinaryOp::Lt, Expr::literal(LiteralValue::Int64(10)));

    assert_eq!(expr.to_string(), "NOT (payload < 10)");
}

#[test]
fn native_expr_can_represent_numeric_cast() {
    let expr = Expr::cast(Expr::column("payload"), DataType::Int64).alias("payload_i64");

    assert_eq!(expr.to_string(), "CAST(payload AS Int64) AS payload_i64");
}

#[test]
fn native_expr_can_represent_phase2_core_expression_surface() {
    let modulo = Expr::binary(Expr::column("payload"), BinaryOp::Modulo, Expr::literal(LiteralValue::Int64(10)));
    let bitwise_and = Expr::binary(Expr::column("flags"), BinaryOp::BitwiseAnd, Expr::literal(LiteralValue::Int64(3)));
    let bitwise_or = Expr::binary(Expr::column("flags"), BinaryOp::BitwiseOr, Expr::literal(LiteralValue::Int64(8)));
    let bitwise_xor = Expr::binary(Expr::column("flags"), BinaryOp::BitwiseXor, Expr::literal(LiteralValue::Int64(1)));
    let shift_left = Expr::binary(Expr::column("flags"), BinaryOp::ShiftLeft, Expr::literal(LiteralValue::Int64(1)));
    let shift_right = Expr::binary(Expr::column("flags"), BinaryOp::ShiftRight, Expr::literal(LiteralValue::Int64(1)));
    let like = Expr::like(Expr::column("name"), "A%");
    let not_like = Expr::not_like_escape(Expr::column("name"), "A!_%", "!");
    let substring = Expr::substring(Expr::column("name"), 2, Some(3));
    let in_list = Expr::in_list(
        Expr::column("status"),
        vec![Expr::literal(LiteralValue::Utf8("O".to_string())), Expr::literal(LiteralValue::Utf8("F".to_string()))],
    );
    let not_in_list = Expr::not_in_list(
        Expr::column("status"),
        vec![Expr::literal(LiteralValue::Utf8("R".to_string())), Expr::literal(LiteralValue::Utf8("X".to_string()))],
    );
    let case = Expr::case(
        vec![
            CaseWhen::new(
                Expr::binary(Expr::column("score"), BinaryOp::GtEq, Expr::literal(LiteralValue::Int64(90))),
                Expr::literal(LiteralValue::Utf8("A".to_string())),
            ),
            CaseWhen::new(
                Expr::binary(Expr::column("score"), BinaryOp::GtEq, Expr::literal(LiteralValue::Int64(80))),
                Expr::literal(LiteralValue::Utf8("B".to_string())),
            ),
        ],
        Expr::literal(LiteralValue::Utf8("C".to_string())),
    );
    let extract = Expr::extract(DatePart::Year, Expr::column("event_ts"));
    let is_null = Expr::is_null(Expr::column("discount"));
    let is_not_null = Expr::is_not_null(Expr::column("discount"));
    let abs = Expr::abs(Expr::column("variance"));
    let round = Expr::round(Expr::column("ratio"), 2, DataType::Float64);
    let timestamp_plus_duration = Expr::binary(
        Expr::column("event_ts"),
        BinaryOp::Add,
        Expr::literal(LiteralValue::Duration { raw: 10, unit: TimeUnit::Nanosecond }),
    );
    let date_plus_interval = Expr::binary(
        Expr::column("shipdate"),
        BinaryOp::Add,
        Expr::literal(LiteralValue::IntervalMonthDayNano { months: 1, days: 2, nanoseconds: 0 }),
    );

    assert_eq!(modulo.to_string(), "(payload % 10)");
    assert_eq!(bitwise_and.to_string(), "(flags & 3)");
    assert_eq!(bitwise_or.to_string(), "(flags | 8)");
    assert_eq!(bitwise_xor.to_string(), "(flags ^ 1)");
    assert_eq!(shift_left.to_string(), "(flags << 1)");
    assert_eq!(shift_right.to_string(), "(flags >> 1)");
    assert_eq!(like.to_string(), "name LIKE \"A%\"");
    assert_eq!(not_like.to_string(), "name NOT LIKE \"A!_%\" ESCAPE \"!\"");
    assert_eq!(substring.to_string(), "SUBSTRING(name FROM 2 FOR 3)");
    assert_eq!(in_list.to_string(), "status IN (\"O\", \"F\")");
    assert_eq!(not_in_list.to_string(), "status NOT IN (\"R\", \"X\")");
    assert_eq!(case.to_string(), "CASE WHEN (score >= 90) THEN \"A\" WHEN (score >= 80) THEN \"B\" ELSE \"C\" END");
    assert_eq!(extract.to_string(), "EXTRACT(YEAR FROM event_ts)");
    assert_eq!(is_null.to_string(), "discount IS NULL");
    assert_eq!(is_not_null.to_string(), "discount IS NOT NULL");
    assert_eq!(abs.to_string(), "ABS(variance)");
    assert_eq!(round.to_string(), "ROUND(ratio, 2)::Float64");
    assert_eq!(timestamp_plus_duration.to_string(), "(event_ts + DURATION(10, Nanosecond))");
    assert_eq!(date_plus_interval.to_string(), "(shipdate + INTERVAL(months=1, days=2, nanoseconds=0))");
}

#[test]
fn native_expr_preserves_qualified_column_identity_separate_from_display_name() {
    let left = ColumnRef::input(0, "id", "left_id");
    let right = ColumnRef::input(1, "id", "right_id");

    assert_ne!(left, right);
    assert_eq!(left.name(), "id");
    assert_eq!(left.display_name(), "left_id");
    assert!(left.is_qualified());
    assert_eq!(Expr::Column(left.clone()).to_string(), "input0.id");
    assert_eq!(Expr::Column(right.clone()).to_string(), "input1.id");
}
