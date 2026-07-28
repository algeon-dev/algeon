//! Errors carry a [`ErrorKind`] variant, an [`ErrorStatus`] for retry
//! semantics, an operation name describing what was being attempted, an
//! optional human-readable message, optional key-value context pairs, and an
//! optional source error. This makes it easy to diagnose failures without
//! losing the underlying cause.

use std::{
    backtrace::{Backtrace, BacktraceStatus},
    error::Error as StdError,
    fmt,
};

use cugraph_sys::{cugraph_error_code_t, cugraph_error_free, cugraph_error_message, cugraph_error_t};

/// Categorisation of cuGraph and library-level errors.
///
/// Each variant describes **what the caller should do**, not which dependency
/// failed. Variants that originate in the raw C API map to the corresponding
/// `CUGRAPH_*` error codes; library-only variants cover Rust-side interop,
/// lifecycle, and I/O boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ErrorKind {
    /// An unexpected or unknown error occurred (`CUGRAPH_UNKNOWN_ERROR`).
    Unexpected,
    /// The resource handle is invalid (`CUGRAPH_INVALID_HANDLE`).
    InvalidHandle,
    /// GPU memory allocation failed (`CUGRAPH_ALLOC_ERROR`).
    OutOfMemory,
    /// An input argument was invalid (`CUGRAPH_INVALID_INPUT`).
    InvalidInput,
    /// An external I/O operation failed.
    IoError,
    /// A library-level precondition, lifecycle contract, or state transition failed.
    InvalidState,
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
            cugraph_error_code_t::CUGRAPH_UNSUPPORTED_TYPE_COMBINATION => ErrorKind::UnsupportedType,
            _ => ErrorKind::Unexpected,
        }
    }

    /// Map from a raw C error-code integer value.
    pub fn from_error_code_raw(raw: u32) -> Self {
        Self::from_error_code(cugraph_error_code_t(raw))
    }

    /// Whether a backtrace should be captured for this error kind.
    ///
    /// Only truly unexpected errors benefit from backtraces; expected
    /// conditions like `InvalidInput` or `NotImplemented` have sufficient
    /// context from the error kind and message alone.
    fn capture_backtrace(&self) -> bool {
        matches!(self, ErrorKind::Unexpected)
    }
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorKind::Unexpected => write!(f, "Unexpected"),
            ErrorKind::InvalidHandle => write!(f, "InvalidHandle"),
            ErrorKind::OutOfMemory => write!(f, "OutOfMemory"),
            ErrorKind::InvalidInput => write!(f, "InvalidInput"),
            ErrorKind::IoError => write!(f, "IoError"),
            ErrorKind::InvalidState => write!(f, "InvalidState"),
            ErrorKind::NotImplemented => write!(f, "NotImplemented"),
            ErrorKind::UnsupportedType => write!(f, "UnsupportedType"),
        }
    }
}

/// Retry semantics for an error.
///
/// Separates retryability from error categorisation so callers can make
/// informed retry decisions without guessing.
///
/// - [`Permanent`](ErrorStatus::Permanent) — will never succeed without
///   external changes (e.g. invalid input, unsupported operation). Do not
///   retry.
/// - [`Temporary`](ErrorStatus::Temporary) — transient, may succeed on retry
///   (e.g. GPU memory pressure, resource contention).
/// - [`Persistent`](ErrorStatus::Persistent) — was transient but still failed
///   after retries. Stop retrying.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorStatus {
    /// The error is permanent — retrying will not help without external
    /// intervention.
    Permanent,
    /// The error is transient — the same operation may succeed if retried.
    Temporary,
    /// The error was transient but persisted after retries — stop retrying.
    Persistent,
}

impl fmt::Display for ErrorStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorStatus::Permanent => write!(f, "permanent"),
            ErrorStatus::Temporary => write!(f, "temporary"),
            ErrorStatus::Persistent => write!(f, "persistent"),
        }
    }
}

/// A cuGraph error with structured context.
///
/// Carries the kind of error, retry status, the operation that was being
/// performed, an optional human-readable message, optional key-value context
/// pairs for debugging, and an optional source error.
///
/// # Display formats
///
/// - **`Display`** (single-line): for structured logs and machine parsing.
///   Format: `{kind} ({status}) at {operation}[ => {message}][, context: { ... }][, source: ...]`
/// - **`Debug`** (multi-line): for human debugging with context breakdown.
/// - **`Debug` alternate** (`{:#?}`): standard Rust struct-style inspection.
pub struct Error {
    kind: ErrorKind,
    status: ErrorStatus,
    operation: &'static str,
    message: String,
    context: Vec<(&'static str, String)>,
    source: Option<Box<dyn StdError + Send + Sync + 'static>>,
    backtrace: Option<Box<Backtrace>>,
}

impl Error {
    /// Create a new error for the given kind and operation.
    ///
    /// The error defaults to [`ErrorStatus::Permanent`]. A backtrace is
    /// captured only for [`ErrorKind::Unexpected`] errors (when the
    /// `RUST_BACKTRACE` environment variable is set).
    pub fn new(kind: ErrorKind, operation: &'static str) -> Self {
        Self {
            kind,
            status: ErrorStatus::Permanent,
            operation,
            message: String::new(),
            context: Vec::new(),
            source: None,
            backtrace: kind
                .capture_backtrace()
                .then(Backtrace::capture)
                .filter(|bt| bt.status() == BacktraceStatus::Captured)
                .map(Box::new),
        }
    }

    /// Set the human-readable error message.
    pub fn with_message(mut self, message: impl Into<String>) -> Self {
        self.message = message.into();
        self
    }

    /// Set the operation, stacking the previous one as context.
    ///
    /// If an operation is already set, it is preserved as
    /// `("called", old_operation)` context, creating a logical call chain
    /// for debugging.
    pub fn with_operation(mut self, operation: &'static str) -> Self {
        if !self.operation.is_empty() {
            self.context.push(("called", self.operation.to_string()));
        }
        self.operation = operation;
        self
    }

    /// Attach a key-value context pair for debugging.
    pub fn with_context(mut self, key: &'static str, value: impl Into<String>) -> Self {
        self.context.push((key, value.into()));
        self
    }

    /// Attach an underlying source error.
    ///
    /// Accepts any source error that is safe to store and report from this
    /// library's public error type.
    pub fn set_source<E>(mut self, source: E) -> Self
    where E: StdError + Send + Sync + 'static {
        debug_assert!(self.source.is_none(), "the source error has been set");
        self.source = Some(Box::new(source));
        self
    }

    /// Mark this error as temporary (retryable).
    pub fn set_temporary(mut self) -> Self {
        self.status = ErrorStatus::Temporary;
        self
    }

    /// Mark this error as permanent (not retryable).
    pub fn set_permanent(mut self) -> Self {
        self.status = ErrorStatus::Permanent;
        self
    }

    /// Mark this error as persistent (was temporary, retries exhausted).
    pub fn set_persistent(mut self) -> Self {
        self.status = ErrorStatus::Persistent;
        self
    }

    /// The kind of this error.
    pub fn kind(&self) -> ErrorKind {
        self.kind
    }

    /// The retry status of this error.
    pub fn status(&self) -> ErrorStatus {
        self.status
    }

    /// Whether this error is temporary (retryable).
    pub fn is_temporary(&self) -> bool {
        self.status == ErrorStatus::Temporary
    }

    /// Whether this error is permanent (not retryable).
    pub fn is_permanent(&self) -> bool {
        self.status == ErrorStatus::Permanent
    }

    /// Whether this error is persistent (transient, but retries are exhausted).
    pub fn is_persistent(&self) -> bool {
        self.status == ErrorStatus::Persistent
    }

    /// The operation that was being performed when the error occurred.
    pub fn operation(&self) -> &'static str {
        self.operation
    }

    /// The human-readable error message, if any.
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if f.alternate() {
            // {:#?} — standard Rust struct-style inspection.
            let mut d = f.debug_struct("Error");
            d.field("kind", &self.kind);
            d.field("status", &self.status);
            d.field("operation", &self.operation);
            if !self.message.is_empty() {
                d.field("message", &self.message);
            }
            if !self.context.is_empty() {
                d.field("context", &self.context);
            }
            if let Some(ref src) = self.source {
                d.field("source", src);
            }
            d.finish()
        } else {
            // {:?} — multi-line format for human debugging.
            write!(f, "{} ({}) at {}", self.kind, self.status, self.operation)?;
            if !self.message.is_empty() {
                write!(f, " => {}", self.message)?;
            }
            if !self.context.is_empty() {
                write!(f, "\n\nContext:")?;
                for (key, value) in &self.context {
                    write!(f, "\n   {}: {}", key, value)?;
                }
            }
            if let Some(ref src) = self.source {
                write!(f, "\n\nSource: {:#}", src)?;
            }
            if let Some(ref bt) = self.backtrace {
                write!(f, "\n\nBacktrace:\n{}", bt)?;
            }
            Ok(())
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({}) at {}", self.kind, self.status, self.operation)?;
        if !self.message.is_empty() {
            write!(f, " => {}", self.message)?;
        }
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
        if let Some(ref src) = self.source {
            write!(f, ", source: {}", src)?;
        }
        Ok(())
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source.as_ref().map(|e| e.as_ref() as &(dyn StdError + 'static))
    }
}

/// Convenience type alias.
pub type Result<T> = std::result::Result<T, Error>;

pub(crate) fn unexpected_null_pointer(operation: &'static str, pointer_name: &'static str) -> Error {
    Error::new(ErrorKind::Unexpected, operation)
        .with_message(format!("C API returned null pointer for {pointer_name}"))
        .with_context("pointer", pointer_name)
}

pub(crate) fn require_non_null<T>(ptr: *mut T, operation: &'static str, pointer_name: &'static str) -> Result<*mut T> {
    if ptr.is_null() { Err(unexpected_null_pointer(operation, pointer_name)) } else { Ok(ptr) }
}

/// Check a C API return code and convert to a Rust `Result`.
///
/// On success (`CUGRAPH_SUCCESS`), returns `Ok(())`. Otherwise extracts the
/// error message from the C error object, frees it, and returns an appropriate
/// [`Error`].
///
/// `OutOfMemory` errors are automatically marked as
/// [`ErrorStatus::Temporary`] since GPU memory pressure is often transient.
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

    let kind = ErrorKind::from_error_code(code);
    let mut err = Error::new(kind, operation);

    // OutOfMemory is typically transient — GPU memory may free up.
    if kind == ErrorKind::OutOfMemory {
        err = err.set_temporary();
    }

    if !error_ptr.is_null() {
        let msg_ptr = cugraph_error_message(error_ptr);
        if !msg_ptr.is_null() {
            let msg = std::ffi::CStr::from_ptr(msg_ptr).to_string_lossy().into_owned();
            err = err.with_message(msg);
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
        assert_eq!(ErrorKind::IoError.to_string(), "IoError");
        assert_eq!(ErrorKind::InvalidState.to_string(), "InvalidState");
        assert_eq!(ErrorKind::NotImplemented.to_string(), "NotImplemented");
        assert_eq!(ErrorKind::UnsupportedType.to_string(), "UnsupportedType");
    }

    #[test]
    fn test_error_status_display() {
        assert_eq!(ErrorStatus::Permanent.to_string(), "permanent");
        assert_eq!(ErrorStatus::Temporary.to_string(), "temporary");
        assert_eq!(ErrorStatus::Persistent.to_string(), "persistent");
    }

    #[test]
    fn test_error_kind_from_error_code() {
        assert_eq!(ErrorKind::from_error_code(cugraph_error_code_t::CUGRAPH_INVALID_HANDLE), ErrorKind::InvalidHandle);
        assert_eq!(ErrorKind::from_error_code(cugraph_error_code_t::CUGRAPH_ALLOC_ERROR), ErrorKind::OutOfMemory);
        assert_eq!(ErrorKind::from_error_code(cugraph_error_code_t::CUGRAPH_INVALID_INPUT), ErrorKind::InvalidInput);
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
    fn test_error_display_with_message_and_context() {
        let err = Error::new(ErrorKind::InvalidInput, "GraphBuilder::build")
            .with_message("src and dst must have the same length")
            .with_context("src_len", "10")
            .with_context("dst_len", "5");

        let display = err.to_string();
        assert_eq!(
            display,
            "InvalidInput (permanent) at GraphBuilder::build \
             => src and dst must have the same length, \
             context: { src_len: 10, dst_len: 5 }"
        );
    }

    #[test]
    fn test_error_display_no_message_no_context() {
        let err = Error::new(ErrorKind::Unexpected, "test_op");
        assert_eq!(err.to_string(), "Unexpected (permanent) at test_op");
    }

    #[test]
    fn test_error_display_message_only() {
        let err = Error::new(ErrorKind::InvalidInput, "test_op").with_message("bad input");
        assert_eq!(err.to_string(), "InvalidInput (permanent) at test_op => bad input");
    }

    #[test]
    fn test_error_display_with_source() {
        let source = std::io::Error::other("GPU fault");
        let err = Error::new(ErrorKind::Unexpected, "ResourceHandle::new").set_source(source);

        let display = err.to_string();
        assert!(display.contains("Unexpected (permanent) at ResourceHandle::new"));
        assert!(display.contains("source: GPU fault"));
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
    fn test_error_debug_multiline() {
        let err = Error::new(ErrorKind::OutOfMemory, "DeviceArray::new")
            .with_message("allocation failed")
            .with_context("size", "1000000");
        let debug = format!("{:?}", err);
        assert!(debug.contains("OutOfMemory (permanent) at DeviceArray::new"));
        assert!(debug.contains("=> allocation failed"));
        assert!(debug.contains("Context:"));
        assert!(debug.contains("size: 1000000"));
    }

    #[test]
    fn test_error_debug_alternate() {
        let err = Error::new(ErrorKind::OutOfMemory, "DeviceArray::new").with_message("allocation failed");
        let debug = format!("{:#?}", err);
        assert!(debug.contains("OutOfMemory"));
        assert!(debug.contains("Permanent"));
        assert!(debug.contains("DeviceArray::new"));
        assert!(debug.contains("allocation failed"));
    }

    #[test]
    fn test_error_accessors() {
        let err = Error::new(ErrorKind::InvalidInput, "test_op").with_message("bad input");
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        assert_eq!(err.operation(), "test_op");
        assert_eq!(err.message(), "bad input");
        assert_eq!(err.status(), ErrorStatus::Permanent);
        assert!(err.is_permanent());
        assert!(!err.is_temporary());
        assert!(!err.is_persistent());
    }

    #[test]
    fn test_error_status_temporary() {
        let err = Error::new(ErrorKind::OutOfMemory, "DeviceArray::new").set_temporary();
        assert!(err.is_temporary());
        assert!(!err.is_permanent());
        assert_eq!(err.status(), ErrorStatus::Temporary);
    }

    #[test]
    fn test_error_status_persistent() {
        let err = Error::new(ErrorKind::OutOfMemory, "DeviceArray::new").set_persistent();
        assert!(!err.is_temporary());
        assert!(!err.is_permanent());
        assert!(err.is_persistent());
        assert_eq!(err.status(), ErrorStatus::Persistent);
    }

    #[test]
    fn test_error_temporary_display() {
        let err = Error::new(ErrorKind::OutOfMemory, "DeviceArray::new").with_message("GPU OOM").set_temporary();
        let display = err.to_string();
        assert!(display.contains("(temporary)"));
        assert!(display.contains("=> GPU OOM"));
    }

    #[test]
    fn test_with_operation_stacking() {
        let err = Error::new(ErrorKind::InvalidInput, "inner_op").with_message("bad data").with_operation("outer_op");

        assert_eq!(err.operation(), "outer_op");
        let display = err.to_string();
        assert!(display.contains("at outer_op"));
        assert!(display.contains("called: inner_op"));
    }

    #[test]
    fn test_check_error_success() {
        // SAFETY: Passing a success code with a null error pointer matches the C API contract;
        // `check_error` will not dereference the pointer on success.
        unsafe {
            let result = check_error(cugraph_error_code_t::CUGRAPH_SUCCESS, std::ptr::null_mut(), "test");
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_require_non_null_rejects_null_pointers() {
        let err = require_non_null::<u8>(std::ptr::null_mut(), "test_op", "result_ptr").expect_err("null must fail");
        assert_eq!(err.kind(), ErrorKind::Unexpected);
        assert_eq!(err.operation(), "test_op");
        assert!(err.message().contains("result_ptr"));
    }

    #[test]
    fn test_check_error_failure_null_ptr() {
        // SAFETY: `check_error` accepts a null error pointer for failure paths and degrades to a
        // synthesized Rust error without dereferencing the pointer.
        unsafe {
            let result = check_error(cugraph_error_code_t::CUGRAPH_INVALID_INPUT, std::ptr::null_mut(), "test_op");
            assert!(result.is_err());
            let err = result.unwrap_err();
            assert_eq!(err.kind(), ErrorKind::InvalidInput);
            assert_eq!(err.operation(), "test_op");
            assert!(err.is_permanent());
        }
    }

    #[test]
    fn test_check_error_failure_unknown_code_maps_to_unexpected() {
        // SAFETY: The test intentionally passes an unknown status code and a null error pointer to
        // verify `check_error`'s fallback mapping logic.
        unsafe {
            let result = check_error(cugraph_error_code_t(777), std::ptr::null_mut(), "test_op");
            assert!(result.is_err());
            let err = result.unwrap_err();
            assert_eq!(err.kind(), ErrorKind::Unexpected);
            assert_eq!(err.operation(), "test_op");
        }
    }

    #[test]
    fn test_check_error_out_of_memory_is_temporary() {
        // SAFETY: The allocation failure code is exercised with a null error pointer to verify
        // Rust-side status mapping; `check_error` handles this without dereferencing the pointer.
        unsafe {
            let result = check_error(cugraph_error_code_t::CUGRAPH_ALLOC_ERROR, std::ptr::null_mut(), "alloc_op");
            assert!(result.is_err());
            let err = result.unwrap_err();
            assert_eq!(err.kind(), ErrorKind::OutOfMemory);
            assert!(err.is_temporary());
        }
    }
}
