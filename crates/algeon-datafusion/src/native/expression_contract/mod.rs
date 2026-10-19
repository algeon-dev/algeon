//! Shared capability and validation contract for DataFusion physical expressions.
//!
//! This module is the single source of truth for native expression admission.
//! It intentionally contains no report formatting. Callers validate first and
//! pass the resulting proof object into lowering.

mod capability;
mod join_residual;
mod rejection;
mod validation;

pub(in crate::native) use capability::{
    ScalarFunctionKind, SupportedBinaryOperator, cast_capability, compatible_types,
    is_decimal128_to_float32_cast_via_float64, is_list_carry_type, is_narrow_decimal_carry, is_numeric_or_decimal_type,
    is_supported_to_timestamp_seconds_cast, is_supported_try_cast_pair, native_aggregate_function_kind,
};
pub(in crate::native) use join_residual::{ValidatedJoinResidual, validate_join_residual};
pub(crate) use rejection::{
    ExpressionCapability, ExpressionRejection, ExpressionRejectionKind, ExpressionRejectionSubject,
};
pub(in crate::native) use validation::{
    MAX_ACTIVATION_NODES, ResolvedColumn, ValidatedPhysicalExpr, binary_operator_subject, case_expr_data_type,
    contains_fallible_operation, declared_decimal_divide_output_type, fallible_operand, is_checked_temporal_arithmetic,
    is_decimal_division, oversized_activation_rejection, physical_expr_data_type, resolve_column,
    validate_physical_expr_contract,
};

#[cfg(test)]
mod tests;
