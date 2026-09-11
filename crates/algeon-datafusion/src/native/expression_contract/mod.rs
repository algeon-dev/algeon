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
    ScalarFunctionKind, SupportedBinaryOperator, cast_capability, compatible_types, is_integral_type,
    is_list_carry_type, is_narrow_decimal_carry, is_numeric_or_decimal_type, is_string_type, is_supported_native_type,
    is_supported_to_timestamp_seconds_cast, is_supported_try_cast_pair, is_timestamp_with_timezone_type,
};
pub(in crate::native) use join_residual::{
    JoinResidualCastMode, ValidatedJoinResidual, join_residual_cast_mode, validate_join_residual,
};
pub(in crate::native) use rejection::{ExpressionCapability, ExpressionRejection, ExpressionRejectionKind};
pub(in crate::native) use validation::{
    ResolvedColumn, ValidatedPhysicalExpr, case_expr_data_type, declared_decimal_divide_output_type,
    physical_expr_data_type, resolve_column, validate_physical_expr_contract as validate_physical_expr,
};

#[cfg(test)]
mod tests;
