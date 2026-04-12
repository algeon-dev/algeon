use rapids_interop::{InteropError, InteropErrorKind};

#[test]
fn interop_error_set_source_keeps_flat_display_and_exposes_source_chain() {
    let cause = std::io::Error::new(std::io::ErrorKind::NotFound, "binding vanished");
    let err = InteropError::new(InteropErrorKind::NotFound, "missing prepared export binding")
        .with_context("buffer_id", "10")
        .set_source(cause);

    assert_eq!(err.to_string(), "missing prepared export binding");
    assert_eq!(
        std::error::Error::source(&err).map(std::string::ToString::to_string).as_deref(),
        Some("binding vanished")
    );
}

#[test]
fn interop_error_equality_ignores_opaque_source_objects() {
    let left = InteropError::new(InteropErrorKind::InvalidView, "same message")
        .with_context("buffer_id", "10")
        .set_source(std::io::Error::other("first source"));
    let right = InteropError::new(InteropErrorKind::InvalidView, "same message")
        .with_context("buffer_id", "10")
        .set_source(std::io::Error::other("second source"));

    assert_eq!(left, right);
}
