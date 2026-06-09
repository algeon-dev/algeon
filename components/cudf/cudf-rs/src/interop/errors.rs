use super::*;

pub(super) fn map_cudf_error(kind: InteropErrorKind, operation: &'static str, error: crate::Error) -> InteropError {
    let mut interop = InteropError::new(kind, error.to_string()).with_context("operation", operation);
    for (key, value) in error.context() {
        interop = interop.with_context(key, value.clone());
    }
    interop
}

pub(super) fn map_interop_error(kind: InteropErrorKind, operation: &'static str, error: InteropError) -> InteropError {
    let mut mapped = InteropError::new(kind, error.to_string()).with_context("operation", operation);
    for (key, value) in error.context() {
        mapped = mapped.with_context(key, value.clone());
    }
    mapped
}
