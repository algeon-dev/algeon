use std::sync::Arc;

use datafusion::{
    logical_expr::{col, lit},
    scalar::ScalarValue,
};
use iceberg::{
    expr::Reference,
    spec::{Datum, NestedField, PrimitiveType, Schema, SchemaRef, Type},
};

use super::filters_to_predicate;

fn schema() -> SchemaRef {
    Arc::new(
        Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
                NestedField::optional(2, "name", Type::Primitive(PrimitiveType::String)).into(),
            ])
            .build()
            .unwrap(),
    )
}

#[test]
fn a_literal_on_the_left_reverses_the_comparison() {
    let predicate = filters_to_predicate(&[lit(5_i64).lt(col("id"))], &schema()).unwrap();
    assert_eq!(predicate, Reference::new("id").greater_than(Datum::long(5)));
}

#[test]
fn an_and_keeps_its_convertible_side_but_an_or_or_not_needs_both() {
    let unconvertible = col("name").like(lit("a%"));
    let id_filter = col("id").gt_eq(lit(10_i64));
    let and = filters_to_predicate(&[id_filter.clone().and(unconvertible.clone())], &schema()).unwrap();
    assert_eq!(and, Reference::new("id").greater_than_or_equal_to(Datum::long(10)));
    assert!(filters_to_predicate(&[id_filter.clone().or(unconvertible.clone())], &schema()).is_none());
    assert!(filters_to_predicate(&[!(id_filter.and(unconvertible))], &schema()).is_none());
}

#[test]
fn a_filter_whose_literal_does_not_bind_is_dropped_and_the_rest_are_conjoined() {
    let unbindable = col("id").eq(lit(ScalarValue::Utf8(Some("not a number".to_string()))));
    let name = col("name").eq(lit(ScalarValue::Utf8View(Some("ann".to_string()))));
    let id = col("id").is_not_null();
    let predicate = filters_to_predicate(&[unbindable, name, id], &schema()).unwrap();
    assert_eq!(
        predicate,
        Reference::new("name").equal_to(Datum::string("ann")).and(Reference::new("id").is_not_null())
    );
}
