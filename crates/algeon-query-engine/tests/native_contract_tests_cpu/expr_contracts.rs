use super::*;

#[test]
fn native_expr_preserves_qualified_column_identity_separate_from_display_name() {
    let left = ColumnRef::input(0, "id", "left_id");
    let right = ColumnRef::input(1, "id", "right_id");

    assert_ne!(left, right);
    assert_eq!(left.name(), "id");
    assert_eq!(left.display_name(), "left_id");
    assert!(left.is_qualified());
    assert_eq!(Expr::Column(left).to_string(), "input0.id");
    assert_eq!(Expr::Column(right).to_string(), "input1.id");
}
