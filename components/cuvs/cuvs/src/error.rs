use std::{
    backtrace::{Backtrace, BacktraceStatus},
    error::Error as StdError,
    ffi::CStr,
    fmt,
};

use cuvs_sys::{cudaError_t, cuvsError_t};

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

    let lower = message.to_ascii_lowercase();
    let mut err = if lower.contains("out of memory") || lower.contains("memory allocation") || lower.contains("alloc") {
        Error::new(ErrorKind::OutOfMemory, operation).set_temporary()
    } else if lower.contains("unsupported") {
        Error::new(ErrorKind::Unsupported, operation)
    } else if lower.contains("invalid")
        || lower.contains("expected")
        || lower.contains("mismatch")
        || lower.contains("must")
        || lower.contains("should")
        || lower.contains("cannot")
    {
        Error::new(ErrorKind::InvalidInput, operation)
    } else {
        Error::new(ErrorKind::Unexpected, operation)
    };

    err = err.with_message(message);
    Err(err)
}

pub(crate) fn check_cuda(code: cudaError_t, operation: &'static str) -> Result<()> {
    if code == cuvs_sys::cudaSuccess {
        return Ok(());
    }

    let message = unsafe {
        let ptr = cuvs_sys::cudaGetErrorString(code);
        if ptr.is_null() {
            format!("CUDA runtime call failed with code {code}")
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };

    let kind = match code {
        1 => ErrorKind::InvalidInput,
        2 => ErrorKind::OutOfMemory,
        3 | 100 => ErrorKind::InvalidState,
        _ => ErrorKind::Unexpected,
    };
    let mut err = Error::new(kind, operation).with_message(message).with_context("cuda_error_code", code.to_string());
    if kind == ErrorKind::OutOfMemory {
        err = err.set_temporary();
    }
    Err(err)
}
