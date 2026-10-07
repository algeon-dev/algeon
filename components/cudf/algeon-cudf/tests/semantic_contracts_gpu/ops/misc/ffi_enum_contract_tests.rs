use cudf::{
    DuplicateKeepOption, NullEquality, NullPolicy, StripSide,
    ops::{binary::BinaryOperator, reduction::ScanType, stream_compaction::NanEquality, unary::UnaryOperator},
};
use cudf_sys::*;

use crate::common::run_named_cases;

fn public_unary_operator_discriminants_match_ffi_contract() {
    let pairs = [
        (UnaryOperator::Sin as i32, CUDF_UNARY_SIN as i32),
        (UnaryOperator::Cos as i32, CUDF_UNARY_COS as i32),
        (UnaryOperator::Tan as i32, CUDF_UNARY_TAN as i32),
        (UnaryOperator::ArcSin as i32, CUDF_UNARY_ARCSIN as i32),
        (UnaryOperator::ArcCos as i32, CUDF_UNARY_ARCCOS as i32),
        (UnaryOperator::ArcTan as i32, CUDF_UNARY_ARCTAN as i32),
        (UnaryOperator::Sinh as i32, CUDF_UNARY_SINH as i32),
        (UnaryOperator::Cosh as i32, CUDF_UNARY_COSH as i32),
        (UnaryOperator::Tanh as i32, CUDF_UNARY_TANH as i32),
        (UnaryOperator::ArcSinh as i32, CUDF_UNARY_ARCSINH as i32),
        (UnaryOperator::ArcCosh as i32, CUDF_UNARY_ARCCOSH as i32),
        (UnaryOperator::ArcTanh as i32, CUDF_UNARY_ARCTANH as i32),
        (UnaryOperator::Exp as i32, CUDF_UNARY_EXP as i32),
        (UnaryOperator::Log as i32, CUDF_UNARY_LOG as i32),
        (UnaryOperator::Sqrt as i32, CUDF_UNARY_SQRT as i32),
        (UnaryOperator::Cbrt as i32, CUDF_UNARY_CBRT as i32),
        (UnaryOperator::Ceil as i32, CUDF_UNARY_CEIL as i32),
        (UnaryOperator::Floor as i32, CUDF_UNARY_FLOOR as i32),
        (UnaryOperator::Abs as i32, CUDF_UNARY_ABS as i32),
        (UnaryOperator::Rint as i32, CUDF_UNARY_RINT as i32),
        (UnaryOperator::BitCount as i32, CUDF_UNARY_BIT_COUNT as i32),
        (UnaryOperator::BitInvert as i32, CUDF_UNARY_BIT_INVERT as i32),
        (UnaryOperator::Not as i32, CUDF_UNARY_NOT as i32),
        (UnaryOperator::Negate as i32, CUDF_UNARY_NEGATE as i32),
    ];

    for (public_value, ffi_value) in pairs {
        assert_eq!(public_value, ffi_value);
    }
}

fn public_binary_operator_discriminants_match_ffi_contract() {
    let pairs = [
        (BinaryOperator::Add as i32, CUDF_BINARY_ADD as i32),
        (BinaryOperator::Sub as i32, CUDF_BINARY_SUB as i32),
        (BinaryOperator::Mul as i32, CUDF_BINARY_MUL as i32),
        (BinaryOperator::Div as i32, CUDF_BINARY_DIV as i32),
        (BinaryOperator::TrueDiv as i32, CUDF_BINARY_TRUE_DIV as i32),
        (BinaryOperator::FloorDiv as i32, CUDF_BINARY_FLOOR_DIV as i32),
        (BinaryOperator::Mod as i32, CUDF_BINARY_MOD as i32),
        (BinaryOperator::Pmod as i32, CUDF_BINARY_PMOD as i32),
        (BinaryOperator::Pymod as i32, CUDF_BINARY_PYMOD as i32),
        (BinaryOperator::Pow as i32, CUDF_BINARY_POW as i32),
        (BinaryOperator::IntPow as i32, CUDF_BINARY_INT_POW as i32),
        (BinaryOperator::LogBase as i32, CUDF_BINARY_LOG_BASE as i32),
        (BinaryOperator::Atan2 as i32, CUDF_BINARY_ATAN2 as i32),
        (BinaryOperator::ShiftLeft as i32, CUDF_BINARY_SHIFT_LEFT as i32),
        (BinaryOperator::ShiftRight as i32, CUDF_BINARY_SHIFT_RIGHT as i32),
        (BinaryOperator::ShiftRightUnsigned as i32, CUDF_BINARY_SHIFT_RIGHT_UNSIGNED as i32),
        (BinaryOperator::BitwiseAnd as i32, CUDF_BINARY_BITWISE_AND as i32),
        (BinaryOperator::BitwiseOr as i32, CUDF_BINARY_BITWISE_OR as i32),
        (BinaryOperator::BitwiseXor as i32, CUDF_BINARY_BITWISE_XOR as i32),
        (BinaryOperator::LogicalAnd as i32, CUDF_BINARY_LOGICAL_AND as i32),
        (BinaryOperator::LogicalOr as i32, CUDF_BINARY_LOGICAL_OR as i32),
        (BinaryOperator::Equal as i32, CUDF_BINARY_EQUAL as i32),
        (BinaryOperator::NotEqual as i32, CUDF_BINARY_NOT_EQUAL as i32),
        (BinaryOperator::Less as i32, CUDF_BINARY_LESS as i32),
        (BinaryOperator::Greater as i32, CUDF_BINARY_GREATER as i32),
        (BinaryOperator::LessEqual as i32, CUDF_BINARY_LESS_EQUAL as i32),
        (BinaryOperator::GreaterEqual as i32, CUDF_BINARY_GREATER_EQUAL as i32),
        (BinaryOperator::NullEquals as i32, CUDF_BINARY_NULL_EQUALS as i32),
        (BinaryOperator::NullNotEquals as i32, CUDF_BINARY_NULL_NOT_EQUALS as i32),
        (BinaryOperator::NullMax as i32, CUDF_BINARY_NULL_MAX as i32),
        (BinaryOperator::NullMin as i32, CUDF_BINARY_NULL_MIN as i32),
        (BinaryOperator::NullLogicalAnd as i32, CUDF_BINARY_NULL_LOGICAL_AND as i32),
        (BinaryOperator::NullLogicalOr as i32, CUDF_BINARY_NULL_LOGICAL_OR as i32),
    ];

    for (public_value, ffi_value) in pairs {
        assert_eq!(public_value, ffi_value);
    }
}

fn public_auxiliary_enum_discriminants_match_ffi_contract() {
    let null_policy_pairs = [
        (NullPolicy::Exclude as i32, CUDF_NULL_POLICY_EXCLUDE as i32),
        (NullPolicy::Include as i32, CUDF_NULL_POLICY_INCLUDE as i32),
    ];
    for (public_value, ffi_value) in null_policy_pairs {
        assert_eq!(public_value, ffi_value);
    }

    let join_null_equality_pairs = [
        (NullEquality::Equal as i32, CUDF_NULL_EQUALITY_EQUAL as i32),
        (NullEquality::Unequal as i32, CUDF_NULL_EQUALITY_UNEQUAL as i32),
    ];
    for (public_value, ffi_value) in join_null_equality_pairs {
        assert_eq!(public_value, ffi_value);
    }

    let stream_compaction_null_equality_pairs = [
        (NullEquality::Equal as i32, CUDF_STREAM_COMPACTION_NULL_EQUALITY_EQUAL as i32),
        (NullEquality::Unequal as i32, CUDF_STREAM_COMPACTION_NULL_EQUALITY_UNEQUAL as i32),
    ];
    for (public_value, ffi_value) in stream_compaction_null_equality_pairs {
        assert_eq!(public_value, ffi_value);
    }

    let strip_side_pairs = [
        (StripSide::Left as i32, CUDF_STRIP_LEFT as i32),
        (StripSide::Right as i32, CUDF_STRIP_RIGHT as i32),
        (StripSide::Both as i32, CUDF_STRIP_BOTH as i32),
    ];
    for (public_value, ffi_value) in strip_side_pairs {
        assert_eq!(public_value, ffi_value);
    }

    let scan_type_pairs = [
        (ScanType::Inclusive as i32, CUDF_SCAN_INCLUSIVE as i32),
        (ScanType::Exclusive as i32, CUDF_SCAN_EXCLUSIVE as i32),
    ];
    for (public_value, ffi_value) in scan_type_pairs {
        assert_eq!(public_value, ffi_value);
    }

    let duplicate_keep_pairs = [
        (DuplicateKeepOption::KeepAny as i32, CUDF_DUPLICATE_KEEP_OPTION_KEEP_ANY as i32),
        (DuplicateKeepOption::KeepFirst as i32, CUDF_DUPLICATE_KEEP_OPTION_KEEP_FIRST as i32),
        (DuplicateKeepOption::KeepLast as i32, CUDF_DUPLICATE_KEEP_OPTION_KEEP_LAST as i32),
        (DuplicateKeepOption::KeepNone as i32, CUDF_DUPLICATE_KEEP_OPTION_KEEP_NONE as i32),
    ];
    for (public_value, ffi_value) in duplicate_keep_pairs {
        assert_eq!(public_value, ffi_value);
    }

    let nan_equality_pairs = [
        (NanEquality::AllEqual as i32, CUDF_STREAM_COMPACTION_NAN_EQUALITY_ALL_EQUAL as i32),
        (NanEquality::Unequal as i32, CUDF_STREAM_COMPACTION_NAN_EQUALITY_UNEQUAL as i32),
    ];
    for (public_value, ffi_value) in nan_equality_pairs {
        assert_eq!(public_value, ffi_value);
    }
}

#[test]
fn ffi_enum_contract_tests_cases() {
    run_named_cases(&[
        (
            "public_unary_operator_discriminants_match_ffi_contract",
            public_unary_operator_discriminants_match_ffi_contract,
        ),
        (
            "public_binary_operator_discriminants_match_ffi_contract",
            public_binary_operator_discriminants_match_ffi_contract,
        ),
        (
            "public_auxiliary_enum_discriminants_match_ffi_contract",
            public_auxiliary_enum_discriminants_match_ffi_contract,
        ),
    ]);
}
