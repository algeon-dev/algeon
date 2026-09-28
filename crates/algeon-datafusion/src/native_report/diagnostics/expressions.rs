use arrow::datatypes::DataType;

use super::{is_supported_arrow_type, stable_identifier};
use crate::native::{ExpressionCapability, ExpressionRejection, ExpressionRejectionKind, ExpressionRejectionSubject};

/// Projects the stable unsupported category for a rejection raised by the
/// shared native expression contract.
pub(super) fn expression_rejection_category(rejection: &ExpressionRejection) -> String {
    match (rejection.kind(), rejection.capability(), rejection.subject()) {
        (
            ExpressionRejectionKind::UnsupportedType,
            ExpressionCapability::CastPair,
            ExpressionRejectionSubject::Cast { source, target },
        ) => cast_type_pair_category("type_cast", source, target),
        (ExpressionRejectionKind::UnsupportedType, _, _) => "type".to_string(),
        (_, ExpressionCapability::TryCastPair, ExpressionRejectionSubject::Cast { source, target }) => {
            cast_type_pair_category("expression_try_cast", source, target)
        },
        (_, ExpressionCapability::Literal, ExpressionRejectionSubject::LiteralType(data_type))
            if !is_supported_arrow_type(data_type) =>
        {
            "expression_literal".to_string()
        },
        (
            _,
            ExpressionCapability::ScalarFunction
            | ExpressionCapability::ScalarFunctionArguments
            | ExpressionCapability::JoinResidualScalarFunction,
            ExpressionRejectionSubject::ScalarFunction(name),
        ) => format!("expression_scalar_function_{}", stable_identifier(name)),
        (_, ExpressionCapability::DynamicFilter, ExpressionRejectionSubject::DynamicFilterError(error)) => {
            format!("expression_dynamic_filter_error_{}", stable_identifier(error).trim_matches('_'))
        },
        // Every operator outside the native binary-operator vocabulary shares one label.
        (_, ExpressionCapability::BinaryOperator, _) => "expression_binary_other".to_string(),
        (_, ExpressionCapability::NegativeExpression, _) => "expression_unary_negative".to_string(),
        (_, ExpressionCapability::IntervalArithmetic, _) => "expression_interval_arithmetic".to_string(),
        (_, ExpressionCapability::LikeCaseInsensitive, _) => "expression_like_case_insensitive".to_string(),
        (_, ExpressionCapability::LikePattern, _) => "expression_like_pattern".to_string(),
        (_, ExpressionCapability::InList, _) => "expression_in_list".to_string(),
        (_, ExpressionCapability::SubstringArguments, _) => "expression_substring_argument".to_string(),
        (_, ExpressionCapability::ExtractField, _) => "expression_extract_field".to_string(),
        _ => "expression".to_string(),
    }
}

pub(super) fn stable_data_type_identifier(data_type: &DataType) -> String {
    stable_identifier(format!("{data_type:?}").as_str()).trim_matches('_').to_string()
}

fn cast_type_pair_category(prefix: &str, source: &DataType, target: &DataType) -> String {
    format!("{prefix}_{}_to_{}", stable_data_type_identifier(source), stable_data_type_identifier(target))
}
