use std::{
    backtrace::{Backtrace, BacktraceStatus},
    error::Error as StdError,
    ffi::CStr,
    fmt,
};

use cudf::interop::contract::{InteropError, InteropErrorKind};
use cuvs_sys::cuvsError_t;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ErrorKind {
    Unexpected,
    InvalidInput,
    InvalidState,
    OutOfMemory,
    Unsupported,
}

impl ErrorKind {
    fn capture_backtrace(&self) -> bool {
        matches!(self, ErrorKind::Unexpected)
    }
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorKind::Unexpected => write!(f, "Unexpected"),
            ErrorKind::InvalidInput => write!(f, "InvalidInput"),
            ErrorKind::InvalidState => write!(f, "InvalidState"),
            ErrorKind::OutOfMemory => write!(f, "OutOfMemory"),
            ErrorKind::Unsupported => write!(f, "Unsupported"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorStatus {
    Permanent,
    Temporary,
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

    pub fn with_message(mut self, message: impl Into<String>) -> Self {
        self.message = message.into();
        self
    }

    pub fn with_operation(mut self, operation: &'static str) -> Self {
        if !self.operation.is_empty() {
            self.context.push(("called", self.operation.to_string()));
        }
        self.operation = operation;
        self
    }

    pub fn with_context(mut self, key: &'static str, value: impl ToString) -> Self {
        self.context.push((key, value.to_string()));
        self
    }

    pub fn set_source<E>(mut self, source: E) -> Self
    where E: StdError + Send + Sync + 'static {
        debug_assert!(self.source.is_none(), "source already set");
        self.source = Some(Box::new(source));
        self
    }

    pub fn set_temporary(mut self) -> Self {
        self.status = ErrorStatus::Temporary;
        self
    }

    pub fn set_permanent(mut self) -> Self {
        self.status = ErrorStatus::Permanent;
        self
    }

    pub fn set_persistent(mut self) -> Self {
        self.status = ErrorStatus::Persistent;
        self
    }

    pub fn kind(&self) -> ErrorKind {
        self.kind
    }

    pub fn status(&self) -> ErrorStatus {
        self.status
    }

    pub fn operation(&self) -> &'static str {
        self.operation
    }

    pub fn message(&self) -> &str {
        &self.message
    }

    /// Structured diagnostic facts forwarded from cudf/interop/native error
    /// contracts. Consumers must not parse [`message`](Self::message) or
    /// [`Display`](fmt::Display) to classify an error.
    pub fn context(&self) -> impl ExactSizeIterator<Item = (&'static str, &str)> {
        self.context.iter().map(|(key, value)| (*key, value.as_str()))
    }

    pub fn context_value(&self, key: &str) -> Option<&str> {
        self.context.iter().rev().find(|(candidate, _)| *candidate == key).map(|(_, value)| value.as_str())
    }

    pub fn is_temporary(&self) -> bool {
        self.status == ErrorStatus::Temporary
    }
}

impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if f.alternate() {
            let mut d = f.debug_struct("Error");
            d.field("kind", &self.kind).field("status", &self.status).field("operation", &self.operation);
            if !self.message.is_empty() {
                d.field("message", &self.message);
            }
            if !self.context.is_empty() {
                d.field("context", &self.context);
            }
            if let Some(source) = &self.source {
                d.field("source", source);
            }
            d.finish()
        } else {
            write!(f, "{} ({}) at {}", self.kind, self.status, self.operation)?;
            if !self.message.is_empty() {
                write!(f, " => {}", self.message)?;
            }
            if !self.context.is_empty() {
                write!(f, "\n\nContext:")?;
                for (key, value) in &self.context {
                    write!(f, "\n   {key}: {value}")?;
                }
            }
            if let Some(source) = &self.source {
                write!(f, "\n\nSource: {source:#}")?;
            }
            if let Some(backtrace) = &self.backtrace {
                write!(f, "\n\nBacktrace:\n{backtrace}")?;
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
                write!(f, " {key}: {value}")?;
            }
            write!(f, " }}")?;
        }
        if let Some(source) = &self.source {
            write!(f, ", source: {source}")?;
        }
        Ok(())
    }
}

impl StdError for Error {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        self.source.as_ref().map(|source| source.as_ref() as &(dyn StdError + 'static))
    }
}

pub type Result<T> = std::result::Result<T, Error>;

pub(crate) fn invalid_input(operation: &'static str, message: impl Into<String>) -> Error {
    Error::new(ErrorKind::InvalidInput, operation).with_message(message)
}

pub(crate) fn invalid_state(operation: &'static str, message: impl Into<String>) -> Error {
    Error::new(ErrorKind::InvalidState, operation).with_message(message)
}

pub(crate) fn unsupported(operation: &'static str, message: impl Into<String>) -> Error {
    Error::new(ErrorKind::Unsupported, operation).with_message(message)
}

pub(crate) fn from_cudf(operation: &'static str, error: cudf::Error) -> Error {
    use cudf::ErrorCode;

    let kind = match error.code() {
        ErrorCode::OutOfMemory | ErrorCode::ForeignResourceExhausted => ErrorKind::OutOfMemory,
        ErrorCode::UnsupportedDtype | ErrorCode::UnsupportedLayout | ErrorCode::UnsupportedOperation => {
            ErrorKind::Unsupported
        },
        ErrorCode::EmptyConstructionRequiresRuntime
        | ErrorCode::RuntimeBindingMismatch
        | ErrorCode::ReadinessUnavailable
        | ErrorCode::ExecutionBindingUnavailable
        | ErrorCode::LeaseAlreadyActive
        | ErrorCode::InvariantViolation
        | ErrorCode::ForeignInvariantViolation => ErrorKind::InvalidState,
        ErrorCode::CudaRuntime
        | ErrorCode::FatalCuda
        | ErrorCode::IoFailure
        | ErrorCode::IoBusy
        | ErrorCode::TransientIoFailure
        | ErrorCode::ForeignTransientDependencyFailure
        | ErrorCode::ForeignPersistentDependencyFailure => ErrorKind::Unexpected,
        _ => ErrorKind::InvalidInput,
    };
    let status = error.status();
    let mut mapped = Error::new(kind, operation)
        .with_message(error.message())
        .with_context("cudf_code", error.code())
        .with_context("cudf_kind", error.kind())
        .with_context("cudf_status", status)
        .with_context("cudf_operation", error.operation());
    mapped = match status {
        cudf::ErrorStatus::Permanent => mapped.set_permanent(),
        cudf::ErrorStatus::Temporary => mapped.set_temporary(),
        cudf::ErrorStatus::Persistent => mapped.set_persistent(),
    };
    mapped.set_source(error)
}

pub(crate) fn from_interop(operation: &'static str, error: InteropError) -> Error {
    let kind = match error.kind() {
        InteropErrorKind::UnsupportedType
        | InteropErrorKind::UnsupportedLayout
        | InteropErrorKind::PlacementUnsupported
        | InteropErrorKind::ProfileUnsupported => ErrorKind::Unsupported,
        InteropErrorKind::ExecutionBindingUnavailable
        | InteropErrorKind::ExecutionBindingMismatch
        | InteropErrorKind::ReadinessUnavailable
        | InteropErrorKind::ReadinessMismatch
        | InteropErrorKind::LeaseAlreadyActive => ErrorKind::InvalidState,
        InteropErrorKind::NotFound
        | InteropErrorKind::NullMaskUnsupported
        | InteropErrorKind::InvalidView
        | InteropErrorKind::MixedDeviceUnsupported => ErrorKind::InvalidInput,
    };
    let mut mapped = Error::new(kind, operation)
        .with_message(error.message())
        .with_context("interop_kind", format!("{:?}", error.kind()));
    for (key, value) in error.context() {
        mapped = mapped.with_context(key, value);
    }
    mapped.set_source(error)
}

pub(crate) fn require_nonzero(value: usize, operation: &'static str, name: &'static str) -> Result<usize> {
    if value == 0 { Err(invalid_input(operation, format!("{name} must be greater than zero"))) } else { Ok(value) }
}

pub(crate) fn check_cuvs(status: cuvsError_t, operation: &'static str) -> Result<()> {
    if status == cuvsError_t::CUVS_SUCCESS {
        return Ok(());
    }

    let message = unsafe {
        let ptr = cuvs_sys::cuvsGetLastErrorText();
        if ptr.is_null() {
            "cuVS C API returned CUVS_ERROR without an error message".to_string()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };

    Err(native_error(status, operation, message))
}

fn native_error(status: cuvsError_t, operation: &'static str, message: impl Into<String>) -> Error {
    Error::new(ErrorKind::Unexpected, operation)
        .with_message(message)
        .with_context("native_status", format!("{status:?}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_message_text_never_changes_error_kind() {
        for message in ["out of memory", "invalid argument", "unsupported dtype", "arbitrary localized native text"] {
            let error = native_error(cuvsError_t::CUVS_ERROR, "test", message);
            assert_eq!(error.kind(), ErrorKind::Unexpected);
            assert_eq!(error.message(), message);
        }
    }
}
