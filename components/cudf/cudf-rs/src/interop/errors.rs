use super::*;

pub(super) fn map_cudf_error(kind: InteropErrorKind, operation: &'static str, error: crate::Error) -> InteropError {
    let kind = if error.kind() == crate::ErrorKind::NotFound { InteropErrorKind::NotFound } else { kind };
    let message = if error.message().is_empty() { error.to_string() } else { error.message().to_string() };
    let mut interop = InteropError::new(kind, message).with_context("operation", operation);
    for (key, value) in error.context() {
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
