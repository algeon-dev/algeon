use crate::core::error::{self, Error, ErrorCode, ErrorKind};

pub(crate) fn unsupported_groupby_validation_error(
    operation: &'static str,
    message: impl Into<String>,
    reason: &'static str,
    remedy: &'static str,
) -> Error {
    Error::new(ErrorKind::InvalidInput, message)
        .with_code(ErrorCode::UnsupportedOperation)
        .with_operation(operation)
        .with_context("reason", reason)
        .with_context("remedy", remedy)
}

pub(crate) fn foreign_groupby_contract_error(
    operation: &'static str,
    message: impl Into<String>,
    resource: &'static str,
    reason: &'static str,
    remedy: &'static str,
) -> Error {
    Error::new(ErrorKind::InvariantViolation, message)
        .with_code(ErrorCode::InvariantViolation)
        .with_operation(operation)
        .with_context("resource", resource)
        .with_context("reason", reason)
        .with_context("remedy", remedy)
}

pub(crate) fn groupby_request_count_limit_error(operation: &'static str, num_requests: usize) -> Error {
    error::limit_validation_error(
        ErrorKind::InvalidInput,
        operation,
        "number of aggregation requests exceeds i32::MAX",
        "aggregation_requests",
        "request_count",
        "i32::MAX",
        num_requests,
        "reduce the number of aggregation requests to fit within libcudf's i32 range",
    )
}

pub(crate) fn groupby_request_agg_count_limit_error(
    operation: &'static str,
    request_index: usize,
    agg_count: usize,
    num_requests: usize,
) -> Error {
    error::limit_validation_error(
        ErrorKind::InvalidInput,
        operation,
        "aggregation count per request exceeds i32::MAX",
        "request_aggregations",
        "aggregation_count",
        "i32::MAX",
        agg_count,
        "reduce the number of aggregations in this request to fit within libcudf's i32 range",
    )
    .with_context("request_index", request_index.to_string())
    .with_context("request_count", num_requests.to_string())
}

pub(crate) fn groupby_total_agg_handle_count_limit_error(
    operation: &'static str,
    total_agg_handles: usize,
    num_requests: usize,
) -> Error {
    error::limit_validation_error(
        ErrorKind::InvalidInput,
        operation,
        "total aggregation handle count exceeds i32::MAX",
        "aggregation_handles",
        "aggregation_handle_count",
        "i32::MAX",
        total_agg_handles,
        "reduce the total number of aggregation handles to fit within libcudf's i32 range",
    )
    .with_context("request_count", num_requests.to_string())
}
