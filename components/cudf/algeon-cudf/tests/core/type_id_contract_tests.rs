use cudf::TypeId;
use cudf_sys::*;

use crate::common::run_named_cases;

fn public_type_id_discriminants_match_ffi_contract() {
    let pairs = [
        (TypeId::Empty as i32, CUDF_TYPE_EMPTY as i32),
        (TypeId::Int8 as i32, CUDF_TYPE_INT8 as i32),
        (TypeId::Int16 as i32, CUDF_TYPE_INT16 as i32),
        (TypeId::Int32 as i32, CUDF_TYPE_INT32 as i32),
        (TypeId::Int64 as i32, CUDF_TYPE_INT64 as i32),
        (TypeId::UInt8 as i32, CUDF_TYPE_UINT8 as i32),
        (TypeId::UInt16 as i32, CUDF_TYPE_UINT16 as i32),
        (TypeId::UInt32 as i32, CUDF_TYPE_UINT32 as i32),
        (TypeId::UInt64 as i32, CUDF_TYPE_UINT64 as i32),
        (TypeId::Float32 as i32, CUDF_TYPE_FLOAT32 as i32),
        (TypeId::Float64 as i32, CUDF_TYPE_FLOAT64 as i32),
        (TypeId::Bool8 as i32, CUDF_TYPE_BOOL8 as i32),
        (TypeId::TimestampDays as i32, CUDF_TYPE_TIMESTAMP_DAYS as i32),
        (TypeId::TimestampSeconds as i32, CUDF_TYPE_TIMESTAMP_SECONDS as i32),
        (TypeId::TimestampMilliseconds as i32, CUDF_TYPE_TIMESTAMP_MILLISECONDS as i32),
        (TypeId::TimestampMicroseconds as i32, CUDF_TYPE_TIMESTAMP_MICROSECONDS as i32),
        (TypeId::TimestampNanoseconds as i32, CUDF_TYPE_TIMESTAMP_NANOSECONDS as i32),
        (TypeId::DurationDays as i32, CUDF_TYPE_DURATION_DAYS as i32),
        (TypeId::DurationSeconds as i32, CUDF_TYPE_DURATION_SECONDS as i32),
        (TypeId::DurationMilliseconds as i32, CUDF_TYPE_DURATION_MILLISECONDS as i32),
        (TypeId::DurationMicroseconds as i32, CUDF_TYPE_DURATION_MICROSECONDS as i32),
        (TypeId::DurationNanoseconds as i32, CUDF_TYPE_DURATION_NANOSECONDS as i32),
        (TypeId::Dictionary32 as i32, CUDF_TYPE_DICTIONARY32 as i32),
        (TypeId::String as i32, CUDF_TYPE_STRING as i32),
        (TypeId::List as i32, CUDF_TYPE_LIST as i32),
        (TypeId::Decimal32 as i32, CUDF_TYPE_DECIMAL32 as i32),
        (TypeId::Decimal64 as i32, CUDF_TYPE_DECIMAL64 as i32),
        (TypeId::Decimal128 as i32, CUDF_TYPE_DECIMAL128 as i32),
        (TypeId::Struct as i32, CUDF_TYPE_STRUCT as i32),
    ];

    for (public_value, ffi_value) in pairs {
        assert_eq!(public_value, ffi_value);
    }
}

fn public_type_id_layout_matches_ffi_layout() {
    assert_eq!(std::mem::size_of::<TypeId>(), std::mem::size_of::<cudf_sys::cudf_type_id_t>());
}

#[test]
fn type_id_contract_tests_cases() {
    run_named_cases(&[
        ("public_type_id_discriminants_match_ffi_contract", public_type_id_discriminants_match_ffi_contract),
        ("public_type_id_layout_matches_ffi_layout", public_type_id_layout_matches_ffi_layout),
    ]);
}
