use super::{InteropError, InteropErrorKind};

pub(super) fn invalid_view_contract_error(
    operation: &'static str,
    message: impl Into<String>,
    resource: &'static str,
    reason: &'static str,
    remedy: impl Into<String>,
) -> InteropError {
    InteropError::new(InteropErrorKind::InvalidView, message)
        .with_context("operation", operation)
        .with_context("resource", resource)
        .with_context("reason", reason)
        .with_context("remedy", remedy.into())
}

pub(super) fn invalid_view_i32_limit_error(operation: &'static str, field: &'static str, value: usize) -> InteropError {
    invalid_view_contract_error(
        operation,
        format!("{field} exceeds i32::MAX"),
        field,
        "exceeds_i32_max",
        format!("reduce {field} to fit within i32::MAX"),
    )
    .with_context(field, value.to_string())
}

pub(super) fn checked_invalid_view_i32_from_usize(
    field: &'static str,
    value: usize,
    operation: &'static str,
) -> Result<i32, InteropError> {
    i32::try_from(value).map_err(|_| invalid_view_i32_limit_error(operation, field, value))
}

pub(super) fn checked_column_index(operation: &'static str, index: usize) -> Result<i32, InteropError> {
    checked_invalid_view_i32_from_usize("column_index", index, operation)
}

pub(super) fn checked_child_index(operation: &'static str, index: usize) -> Result<i32, InteropError> {
    checked_invalid_view_i32_from_usize("child_index", index, operation)
}

pub(super) fn map_cudf_error(kind: InteropErrorKind, operation: &'static str, error: crate::Error) -> InteropError {
    let kind = if error.kind() == crate::ErrorKind::NotFound { InteropErrorKind::NotFound } else { kind };
    let message = if error.message().is_empty() { error.to_string() } else { error.message().to_string() };
    let mut interop = InteropError::new(kind, message).with_context("operation", operation);
    for (key, value) in error.details() {
        interop = interop.with_context(key, value.clone());
    }
    interop.set_source(error)
}

pub(super) fn map_interop_error(kind: InteropErrorKind, operation: &'static str, error: InteropError) -> InteropError {
    let kind = if error.kind() == InteropErrorKind::NotFound { InteropErrorKind::NotFound } else { kind };
    let message = error.message().to_string();
    let context = error.context().iter().map(|(key, value)| (*key, value.clone())).collect::<Vec<_>>();
    let mut mapped = InteropError::new(kind, message).with_context("operation", operation);
    for (key, value) in context {
        mapped = mapped.with_context(key, value);
    }
    mapped.set_source(error)
}
