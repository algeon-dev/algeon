use super::*;

#[test]
fn decimal128_type_support_policy_accepts_valid_arrow_decimals() {
    let decimal_with_fractional_scale = DataType::Decimal128(38, 2);
    let decimal_with_negative_scale = DataType::Decimal128(20, -1);

    assert!(is_supported_arrow_type(&decimal_with_fractional_scale));
    assert_eq!(
        cudf_dtype_from_arrow_type(&decimal_with_fractional_scale),
        Some(CudfDataType::fixed_point(TypeId::Decimal128, -2))
    );
    assert!(is_supported_arrow_type(&decimal_with_negative_scale));
    assert_eq!(
        cudf_dtype_from_arrow_type(&decimal_with_negative_scale),
        Some(CudfDataType::fixed_point(TypeId::Decimal128, 1))
    );
}

#[test]
fn decimal128_type_support_policy_rejects_invalid_arrow_decimals() {
    let invalid_precision = DataType::Decimal128(39, 2);
    let invalid_scale = DataType::Decimal128(10, 11);

    assert!(!is_supported_arrow_type(&invalid_precision));
    assert_eq!(cudf_dtype_from_arrow_type(&invalid_precision), None);
    assert!(!is_supported_arrow_type(&invalid_scale));
    assert_eq!(cudf_dtype_from_arrow_type(&invalid_scale), None);
}

#[test]
fn decimal128_cast_support_uses_type_support_policy() {
    let valid_source = DataType::Decimal128(20, -1);
    let valid_target = DataType::Decimal128(38, 2);
    let float64_target = DataType::Float64;
    let float64_source = DataType::Float64;
    let int32_source = DataType::Int32;
    let int32_target = DataType::Int32;
    let scale_zero_decimal = DataType::Decimal128(21, 0);
    let fractional_decimal = DataType::Decimal128(21, 6);
    let invalid_decimal = DataType::Decimal128(39, 2);

    assert!(is_supported_arrow_cast(&valid_source, &valid_target));
    assert!(is_supported_arrow_cast(&valid_source, &float64_target));
    assert!(is_supported_arrow_cast(&float64_source, &valid_target));
    assert!(is_supported_arrow_cast(&int32_source, &valid_target));
    assert!(is_supported_arrow_cast(&scale_zero_decimal, &int32_target));
    assert!(is_supported_arrow_implicit_cast(&valid_source, &valid_target));
    assert!(!is_supported_arrow_implicit_cast(&valid_source, &float64_target));
    assert!(!is_supported_arrow_implicit_cast(&float64_source, &valid_target));
    assert!(!is_supported_arrow_implicit_cast(&scale_zero_decimal, &int32_target));
    assert!(!is_supported_arrow_cast(&fractional_decimal, &int32_target));
    assert!(!is_supported_arrow_cast(&invalid_decimal, &invalid_decimal));
    assert!(!is_supported_arrow_cast(&invalid_decimal, &valid_target));
    assert!(!is_supported_arrow_cast(&invalid_decimal, &float64_target));
    assert!(!is_supported_arrow_implicit_cast(&invalid_decimal, &float64_target));
    assert!(!is_supported_arrow_cast(&valid_source, &invalid_decimal));
    assert!(!is_supported_arrow_cast(&float64_source, &invalid_decimal));
}
