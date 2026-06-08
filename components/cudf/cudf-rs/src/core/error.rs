//! Errors carry structured context — kind, status, operation, key-value pairs,
//! and an optional source — so that GPU pipeline failures are debuggable without
//! guesswork.

use std::{ffi::CStr, fmt};

use rapids_interop::{InteropError, InteropErrorKind};

/// The kind of error that occurred.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorKind {
    /// The caller passed an invalid value or combination of values.
    InvalidArgument,
    /// The requested operation or type combination is not supported.
    NotSupported,
    /// A logical precondition or internal contract was violated.
    InvalidOperation,
    /// A CUDA runtime error occurred.
    CudaError,
    /// A fatal CUDA error occurred (device may be unusable).
    FatalCudaError,
    /// Device or host memory allocation failed.
    OutOfMemory,
    /// An I/O error occurred (file not found, permission denied, etc.).
    IoError,
    /// An operation was attempted on an unsupported data type.
    TypeError,
    /// A null pointer or null value was encountered unexpectedly.
    NullError,
    /// An index was out of bounds.
    IndexError,
    /// An unknown or unclassified error.
    Unknown,
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorKind::InvalidArgument => write!(f, "InvalidArgument"),
            ErrorKind::NotSupported => write!(f, "NotSupported"),
            ErrorKind::InvalidOperation => write!(f, "InvalidOperation"),
            ErrorKind::CudaError => write!(f, "CudaError"),
            ErrorKind::FatalCudaError => write!(f, "FatalCudaError"),
            ErrorKind::OutOfMemory => write!(f, "OutOfMemory"),
            ErrorKind::IoError => write!(f, "IoError"),
            ErrorKind::TypeError => write!(f, "TypeError"),
            ErrorKind::NullError => write!(f, "NullError"),
            ErrorKind::IndexError => write!(f, "IndexError"),
            ErrorKind::Unknown => write!(f, "Unknown"),
        }
    }
}

/// Whether an error is expected to be transient or permanent.
///
/// This classification helps callers decide whether retrying an operation
/// makes sense (e.g., a CUDA OOM may clear after other work completes,
/// but a type mismatch will never succeed).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorStatus {
    /// The error is caused by invalid input or logic and will never succeed
    /// if retried with the same arguments.
    Permanent,
    /// The error may be transient (e.g., CUDA OOM, device busy) and retrying
    /// could succeed.
    Temporary,
    /// The error has been observed repeatedly and is unlikely to resolve
    /// without intervention, but is not strictly a logic error.
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

/// An error from a cudf operation.
///
/// - `kind` — what category of failure occurred
/// - `status` — permanent / temporary / persistent
/// - `operation` — which API entry point was executing
/// - `context` — arbitrary key-value pairs (dtype, column name, …)
/// - `source` — an optional chained cause
pub struct Error {
    kind: ErrorKind,
    message: String,
    status: ErrorStatus,
    operation: &'static str,
    context: Vec<(&'static str, String)>,
    source: Option<anyhow::Error>,
}

impl Error {
    /// Create a new error with the given kind and message.
    ///
    /// Defaults: `Permanent` status, empty operation, no context, no source.
    /// This keeps all existing call sites compiling unchanged.
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            status: ErrorStatus::Permanent,
            operation: "",
            context: Vec::new(),
            source: None,
        }
    }

    // -- Builder methods (consuming self, for chaining) ----------------------

    /// Set the operation that was being performed when the error occurred.
    pub fn with_operation(mut self, operation: &'static str) -> Self {
        self.operation = operation;
        self
    }

    /// Append a key-value context pair.
    pub fn with_context(mut self, key: &'static str, value: impl Into<String>) -> Self {
        self.context.push((key, value.into()));
        self
    }

    /// Attach a source error.
    pub fn set_source(mut self, src: impl Into<anyhow::Error>) -> Self {
        self.source = Some(src.into());
        self
    }

    /// Mark this error as temporary (transient, retryable).
    pub fn set_temporary(mut self) -> Self {
        self.status = ErrorStatus::Temporary;
        self
    }

    /// Mark this error as permanent (will never succeed if retried).
    pub fn set_permanent(mut self) -> Self {
        self.status = ErrorStatus::Permanent;
        self
    }

    /// Mark this error as persistent (repeated failure, needs intervention).
    pub fn set_persistent(mut self) -> Self {
        self.status = ErrorStatus::Persistent;
        self
    }

    // -- Accessors -----------------------------------------------------------

    /// Returns the kind of this error.
    pub fn kind(&self) -> ErrorKind {
        self.kind
    }

    /// Returns the error message.
    pub fn message(&self) -> &str {
        &self.message
    }

    /// Returns the error status (permanent, temporary, or persistent).
    pub fn status(&self) -> ErrorStatus {
        self.status
    }

    /// Returns the operation that was being performed.
    pub fn operation(&self) -> &'static str {
        self.operation
    }

    /// Returns the context key-value pairs.
    pub fn context(&self) -> &[(&'static str, String)] {
        &self.context
    }

    /// Returns `true` if the error status is `Temporary`.
    pub fn is_temporary(&self) -> bool {
        self.status == ErrorStatus::Temporary
    }

    /// Returns `true` if the error status is `Permanent`.
    pub fn is_permanent(&self) -> bool {
        self.status == ErrorStatus::Permanent
    }

    /// Returns `true` if the error status is `Persistent`.
    pub fn is_persistent(&self) -> bool {
        self.status == ErrorStatus::Persistent
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({})", self.kind, self.status)?;

        if !self.operation.is_empty() {
            write!(f, " at {}", self.operation)?;
        }

        if !self.context.is_empty() {
            write!(f, ", context: {{ ")?;
            for (i, (k, v)) in self.context.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}: {}", k, v)?;
            }
            write!(f, " }}")?;
        }

        if !self.message.is_empty() {
            write!(f, " => {}", self.message)?;
        }

        if let Some(ref src) = self.source {
            write!(f, ", source: {}", src)?;
        }

        Ok(())
    }
}

impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{} ({}):", self.kind, self.status)?;

        if !self.operation.is_empty() {
            writeln!(f, "  operation: {}", self.operation)?;
        }

        if !self.context.is_empty() {
            writeln!(f, "  context:")?;
            for (k, v) in &self.context {
                writeln!(f, "    {}: {}", k, v)?;
            }
        }

        if !self.message.is_empty() {
            writeln!(f, "  message: {}", self.message)?;
        }

        if let Some(ref src) = self.source {
            writeln!(f, "  source: {:?}", src)?;
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

/// Classify an [`ErrorKind`] as temporary or permanent.
///
/// CUDA runtime errors and OOM are considered temporary (the GPU may recover),
/// while logic errors (type mismatch, invalid args) are permanent.
fn default_status_for_kind(kind: ErrorKind) -> ErrorStatus {
    match kind {
        ErrorKind::CudaError | ErrorKind::OutOfMemory => ErrorStatus::Temporary,
        ErrorKind::FatalCudaError => ErrorStatus::Persistent,
        _ => ErrorStatus::Permanent,
    }
}

fn map_interop_error_kind(kind: InteropErrorKind) -> ErrorKind {
    match kind {
        InteropErrorKind::UnsupportedType => ErrorKind::TypeError,
        InteropErrorKind::NullMaskUnsupported => ErrorKind::NullError,
        InteropErrorKind::ExecutionBindingUnavailable | InteropErrorKind::ReadinessUnavailable => ErrorKind::CudaError,
        InteropErrorKind::UnsupportedLayout
        | InteropErrorKind::AmbientInteropUnsupported
        | InteropErrorKind::PlacementUnsupported
        | InteropErrorKind::MixedDeviceUnsupported
        | InteropErrorKind::ProfileUnsupported => ErrorKind::NotSupported,
        InteropErrorKind::ExecutionBindingMismatch
        | InteropErrorKind::InvalidView
        | InteropErrorKind::ReadinessMismatch
        | InteropErrorKind::LeaseAlreadyActive => ErrorKind::InvalidOperation,
    }
}

impl From<InteropError> for Error {
    fn from(error: InteropError) -> Self {
        let kind = map_interop_error_kind(error.kind());
        let message = error.to_string();
        let context = error.context().iter().map(|(key, value)| (*key, value.clone())).collect::<Vec<_>>();

        let mut mapped = Error::new(kind, message).set_source(error);
        mapped.status = default_status_for_kind(kind);
        mapped.context = context;
        mapped
    }
}

/// Convert a C shim status code into a `Result<()>`.
///
/// On success (`CUDF_SUCCESS`), returns `Ok(())`.
/// On failure, reads the thread-local error message and returns an `Err`
/// with the error status automatically classified by [`default_status_for_kind`].
pub(crate) fn check_status(status: cudf_sys::cudf_status_t) -> Result<()> {
    if status == cudf_sys::CUDF_SUCCESS {
        return Ok(());
    }

    let kind = match status {
        cudf_sys::CUDF_ERROR_INVALID_OPERATION => ErrorKind::InvalidOperation,
        cudf_sys::CUDF_ERROR_CUDA => ErrorKind::CudaError,
        cudf_sys::CUDF_ERROR_FATAL_CUDA => ErrorKind::FatalCudaError,
        cudf_sys::CUDF_ERROR_OUT_OF_MEMORY => ErrorKind::OutOfMemory,
        cudf_sys::CUDF_ERROR_IO => ErrorKind::IoError,
        cudf_sys::CUDF_ERROR_TYPE => ErrorKind::TypeError,
        cudf_sys::CUDF_ERROR_NULL => ErrorKind::NullError,
        cudf_sys::CUDF_ERROR_INDEX => ErrorKind::IndexError,
        _ => ErrorKind::Unknown,
    };

    // SAFETY: cudf_last_error_message returns either null or a pointer to a
    // thread-local static C string that remains valid until the next FFI call
    // on this thread. We immediately copy that string into an owned String
    // below, so no intervening FFI call can invalidate the borrowed pointer
    // before we finish reading it. CStr::from_ptr still requires a valid
    // null-terminated string for the duration of this immediate copy.
    let message = unsafe {
        let ptr = cudf_sys::cudf_last_error_message();
        if ptr.is_null() {
            String::from("(no error message)")
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };

    let mut err = Error::new(kind, message);
    err.status = default_status_for_kind(kind);
    Err(err)
}

pub(crate) fn check_status_in(status: cudf_sys::cudf_status_t, operation: &'static str) -> Result<()> {
    check_status(status).map_err(|e| e.with_operation(operation))
}

pub(crate) fn check_status_in_with<I>(
    status: cudf_sys::cudf_status_t,
    operation: &'static str,
    context: I,
) -> Result<()>
where
    I: IntoIterator<Item = (&'static str, String)>,
{
    check_status_in(status, operation)
        .map_err(|err| context.into_iter().fold(err, |acc, (k, v)| acc.with_context(k, v)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_status_cuda_is_temporary() {
        assert_eq!(default_status_for_kind(ErrorKind::CudaError), ErrorStatus::Temporary);
        assert_eq!(default_status_for_kind(ErrorKind::OutOfMemory), ErrorStatus::Temporary);
    }

    #[test]
    fn default_status_fatal_cuda_is_persistent() {
        assert_eq!(default_status_for_kind(ErrorKind::FatalCudaError), ErrorStatus::Persistent);
    }
}
