use cudf::scalar::Scalar;

#[test]
fn test_string_scalar_value_round_trip() {
    let scalar = Scalar::from_string("hello").unwrap();
    assert_eq!(scalar.string_value().unwrap(), "hello");
}

#[test]
fn test_empty_string_scalar_value_round_trip() {
    let scalar = Scalar::from_string("").unwrap();
    assert_eq!(scalar.string_value().unwrap(), "");
}
