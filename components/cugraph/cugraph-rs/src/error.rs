//! Errors carry a [`ErrorKind`] variant, an operation name describing what was
//! being attempted, optional key-value context pairs, and an optional source
//! error. This makes it easy to diagnose failures without losing the
//! underlying cause.

use std::fmt;

use cugraph_sys::{
    cugraph_error_code_t, cugraph_error_free, cugraph_error_message, cugraph_error_t,
};

/// Categorisation of cuGraph errors.
///
/// Each variant maps to a corresponding `CUGRAPH_*` error code from the C API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorKind {
    /// An unexpected or unknown error occurred (`CUGRAPH_UNKNOWN_ERROR`).
    Unexpected,
    /// The resource handle is invalid (`CUGRAPH_INVALID_HANDLE`).
    InvalidHandle,
    /// GPU memory allocation failed (`CUGRAPH_ALLOC_ERROR`).
    OutOfMemory,
    /// An input argument was invalid (`CUGRAPH_INVALID_INPUT`).
    InvalidInput,
    /// The requested operation is not implemented (`CUGRAPH_NOT_IMPLEMENTED`).
    NotImplemented,
    /// The combination of types is not supported (`CUGRAPH_UNSUPPORTED_TYPE_COMBINATION`).
    UnsupportedType,
}

impl ErrorKind {
    /// Map a C error code to an [`ErrorKind`].
    pub fn from_error_code(code: cugraph_error_code_t) -> Self {
        match code {
            cugraph_error_code_t::CUGRAPH_INVALID_HANDLE => ErrorKind::InvalidHandle,
            cugraph_error_code_t::CUGRAPH_ALLOC_ERROR => ErrorKind::OutOfMemory,
            cugraph_error_code_t::CUGRAPH_INVALID_INPUT => ErrorKind::InvalidInput,
            cugraph_error_code_t::CUGRAPH_NOT_IMPLEMENTED => ErrorKind::NotImplemented,
            cugraph_error_code_t::CUGRAPH_UNSUPPORTED_TYPE_COMBINATION => {
                ErrorKind::UnsupportedType
            }
            _ => ErrorKind::Unexpected,
        }
    }

    /// Map from a raw C error-code integer value.
    pub fn from_error_code_raw(raw: u32) -> Self {
        Self::from_error_code(cugraph_error_code_t(raw))
    }
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorKind::Unexpected => write!(f, "Unexpected"),
            ErrorKind::InvalidHandle => write!(f, "InvalidHandle"),
            ErrorKind::OutOfMemory => write!(f, "OutOfMemory"),
            ErrorKind::InvalidInput => write!(f, "InvalidInput"),
            ErrorKind::NotImplemented => write!(f, "NotImplemented"),
            ErrorKind::UnsupportedType => write!(f, "UnsupportedType"),
        }
    }
}

/// A cuGraph error with structured context.
///
/// Carries the kind of error, the operation that was being performed, optional
/// key-value context pairs for debugging, and an optional source error.
pub struct Error {
    kind: ErrorKind,
    operation: &'static str,
    context: Vec<(&'static str, String)>,
    source: Option<anyhow::Error>,
}

impl Error {
    /// Create a new error for the given kind and operation.
    pub fn new(kind: ErrorKind, operation: &'static str) -> Self {
        Self {
            kind,
            operation,
            context: Vec::new(),
            source: None,
        }
    }

    /// Attach a key-value context pair for debugging.
    pub fn with_context(mut self, key: &'static str, value: impl Into<String>) -> Self {
        self.context.push((key, value.into()));
        self
    }

    /// Attach an underlying source error.
    ///
    /// Accepts any type that converts into [`anyhow::Error`], which covers all
    /// `std::error::Error + Send + Sync + 'static` types as well as bare
    /// `anyhow::Error` values.
    pub fn set_source(mut self, source: impl Into<anyhow::Error>) -> Self {
        debug_assert!(self.source.is_none(), "the source error has been set");
        self.source = Some(source.into());
        self
    }

    /// The kind of this error.
    pub fn kind(&self) -> ErrorKind {
        self.kind
    }

    /// The operation that was being performed when the error occurred.
    pub fn operation(&self) -> &'static str {
        self.operation
    }
}

impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut d = f.debug_struct("Error");
        d.field("kind", &self.kind);
        d.field("operation", &self.operation);
        if !self.context.is_empty() {
            d.field("context", &self.context);
        }
        if let Some(ref src) = self.source {
            d.field("source", src);
        }
        d.finish()
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({}", self.kind, self.operation)?;
        if !self.context.is_empty() {
            write!(f, ", context: {{")?;
            for (i, (key, value)) in self.context.iter().enumerate() {
                if i > 0 {
                    write!(f, ",")?;
                }
                write!(f, " {}: {}", key, value)?;
            }
            write!(f, " }}")?;
        }
        write!(f, ")")?;
        if let Some(ref src) = self.source {
            write!(f, " => {}", src)?;
        }
        Ok(())
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source.as_ref().map(|e| e.as_ref())
    }
}

/// Convenience type alias.
pub type Result<T> = std::result::Result<T, Error>;

/// Check a C API return code and convert to a Rust `Result`.
///
/// On success (`CUGRAPH_SUCCESS`), returns `Ok(())`. Otherwise extracts the
/// error message from the C error object, frees it, and returns an appropriate
/// [`Error`].
///
/// # Safety
///
/// `error_ptr` must be a valid pointer to a `*mut cugraph_error_t` that was
/// populated by the C API call. This function is intended to be called
/// immediately after a C API call in unsafe blocks within this crate.
pub(crate) unsafe fn check_error(
    code: cugraph_error_code_t,
    error_ptr: *mut cugraph_error_t,
    operation: &'static str,
) -> Result<()> {
    if code == cugraph_error_code_t::CUGRAPH_SUCCESS {
        return Ok(());
    }

    let kind = ErrorKind::from_error_code_raw(code.0);
    let mut err = Error::new(kind, operation);

    if !error_ptr.is_null() {
        let msg_ptr = cugraph_error_message(error_ptr);
        if !msg_ptr.is_null() {
            let msg = std::ffi::CStr::from_ptr(msg_ptr)
                .to_string_lossy()
                .into_owned();
            err = err.with_context("message", msg);
        }
        cugraph_error_free(error_ptr);
    }

    Err(err)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_kind_display() {
        assert_eq!(ErrorKind::Unexpected.to_string(), "Unexpected");
        assert_eq!(ErrorKind::InvalidHandle.to_string(), "InvalidHandle");
        assert_eq!(ErrorKind::OutOfMemory.to_string(), "OutOfMemory");
        assert_eq!(ErrorKind::InvalidInput.to_string(), "InvalidInput");
        assert_eq!(ErrorKind::NotImplemented.to_string(), "NotImplemented");
        assert_eq!(ErrorKind::UnsupportedType.to_string(), "UnsupportedType");
    }

    #[test]
    fn test_error_kind_from_error_code() {
        assert_eq!(
            ErrorKind::from_error_code(cugraph_error_code_t::CUGRAPH_INVALID_HANDLE),
            ErrorKind::InvalidHandle
        );
        assert_eq!(
            ErrorKind::from_error_code(cugraph_error_code_t::CUGRAPH_ALLOC_ERROR),
            ErrorKind::OutOfMemory
        );
        assert_eq!(
            ErrorKind::from_error_code(cugraph_error_code_t::CUGRAPH_INVALID_INPUT),
            ErrorKind::InvalidInput
        );
        assert_eq!(
            ErrorKind::from_error_code(cugraph_error_code_t::CUGRAPH_NOT_IMPLEMENTED),
            ErrorKind::NotImplemented
        );
        assert_eq!(
            ErrorKind::from_error_code(cugraph_error_code_t::CUGRAPH_UNSUPPORTED_TYPE_COMBINATION),
            ErrorKind::UnsupportedType
        );
    }

    #[test]
    fn test_error_kind_from_unknown_raw_error_code() {
        assert_eq!(ErrorKind::from_error_code_raw(999), ErrorKind::Unexpected);
    }

    #[test]
    fn test_error_display_with_context() {
        let err = Error::new(ErrorKind::InvalidInput, "Graph::from_edge_list")
            .with_context("message", "src and dst must have the same length")
            .with_context("src_len", "10")
            .with_context("dst_len", "5");

        let display = err.to_string();
        assert_eq!(
            display,
            "InvalidInput (Graph::from_edge_list, \
             context: { message: src and dst must have the same length, \
             src_len: 10, dst_len: 5 })"
        );
    }

    #[test]
    fn test_error_display_with_source() {
        let source = std::io::Error::other("GPU fault");
        let err = Error::new(ErrorKind::Unexpected, "ResourceHandle::new").set_source(source);

        let display = err.to_string();
        assert!(display.contains("Unexpected (ResourceHandle::new)"));
        assert!(display.contains("=> GPU fault"));
    }

    #[test]
    fn test_error_source_chain() {
        let source = std::io::Error::other("GPU fault");
        let err = Error::new(ErrorKind::Unexpected, "ResourceHandle::new").set_source(source);

        let std_source = std::error::Error::source(&err);
        assert!(std_source.is_some());
        assert_eq!(std_source.unwrap().to_string(), "GPU fault");
    }

    #[test]
    fn test_error_debug() {
        let err =
            Error::new(ErrorKind::OutOfMemory, "DeviceArray::new").with_context("size", "1000000");
        let debug = format!("{:?}", err);
        assert!(debug.contains("OutOfMemory"));
        assert!(debug.contains("DeviceArray::new"));
    }

    #[test]
    fn test_error_accessors() {
        let err = Error::new(ErrorKind::InvalidInput, "test_op");
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        assert_eq!(err.operation(), "test_op");
    }

    #[test]
    fn test_check_error_success() {
        unsafe {
            let result = check_error(
                cugraph_error_code_t::CUGRAPH_SUCCESS,
                std::ptr::null_mut(),
                "test",
            );
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_check_error_failure_null_ptr() {
        unsafe {
            let result = check_error(
                cugraph_error_code_t::CUGRAPH_INVALID_INPUT,
                std::ptr::null_mut(),
                "test_op",
            );
            assert!(result.is_err());
            let err = result.unwrap_err();
            assert_eq!(err.kind(), ErrorKind::InvalidInput);
            assert_eq!(err.operation(), "test_op");
        }
    }

    #[test]
    fn test_check_error_failure_unknown_code_maps_to_unexpected() {
        unsafe {
            let result = check_error(cugraph_error_code_t(777), std::ptr::null_mut(), "test_op");
            assert!(result.is_err());
            let err = result.unwrap_err();
            assert_eq!(err.kind(), ErrorKind::Unexpected);
            assert_eq!(err.operation(), "test_op");
        }
    }
}
