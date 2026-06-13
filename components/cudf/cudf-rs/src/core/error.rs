//! Errors carry structured context — kind, code, status, operation, key-value pairs,
//! and an optional source — so that GPU pipeline failures are debuggable without
//! guesswork.

use std::{borrow::Cow, error::Error as StdError, ffi::CStr, fmt, ptr::NonNull};

use rapids_interop::{InteropError, InteropErrorKind};

type BoxedSourceError = Box<dyn StdError + Send + Sync + 'static>;

/// The kind of error that occurred.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ErrorKind {
    /// The caller provided invalid values, schema requests, or foreign inputs.
    InvalidInput,
    /// The caller referenced a required named resource that does not exist.
    NotFound,
    /// The requested dtype, layout, or operation is not supported by this path.
    Unsupported,
    /// A runtime, readiness, provenance, or execution binding contract failed.
    ExecutionContext,
    /// The operation ran out of memory or another bounded resource.
    ResourceExhausted,
    /// A lower-layer CUDA, IO, or foreign dependency failed.
    DependencyFailure,
    /// The library or a foreign backend returned impossible contract metadata.
    InvariantViolation,
}

/// A stable, machine-readable error sub-code within an [`ErrorKind`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ErrorCode {
    InvalidInput,
    NotFound,
    Unsupported,
    ExecutionContext,
    ResourceExhausted,
    DependencyFailure,
    InvariantViolation,
    MissingResource,
    MissingColumn,
    DuplicateColumnName,
    DuplicateKey,
    ColumnLengthMismatch,
    TypeMismatch,
    IndexOutOfBounds,
    EmptyConstructionRequiresRuntime,
    CrossDeviceInputs,
    CrossProvenanceInputs,
    RuntimeBindingMismatch,
    ReadinessUnavailable,
    ExecutionBindingUnavailable,
    LeaseAlreadyActive,
    UnsupportedDtype,
    UnsupportedLayout,
    UnsupportedOperation,
    OutOfMemory,
    CudaRuntime,
    FatalCuda,
    IoFailure,
    NullUnexpected,
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorCode::InvalidInput => write!(f, "InvalidInput"),
            ErrorCode::NotFound => write!(f, "NotFound"),
            ErrorCode::Unsupported => write!(f, "Unsupported"),
            ErrorCode::ExecutionContext => write!(f, "ExecutionContext"),
            ErrorCode::ResourceExhausted => write!(f, "ResourceExhausted"),
            ErrorCode::DependencyFailure => write!(f, "DependencyFailure"),
            ErrorCode::InvariantViolation => write!(f, "InvariantViolation"),
            ErrorCode::MissingResource => write!(f, "MissingResource"),
            ErrorCode::MissingColumn => write!(f, "MissingColumn"),
            ErrorCode::DuplicateColumnName => write!(f, "DuplicateColumnName"),
            ErrorCode::DuplicateKey => write!(f, "DuplicateKey"),
            ErrorCode::ColumnLengthMismatch => write!(f, "ColumnLengthMismatch"),
            ErrorCode::TypeMismatch => write!(f, "TypeMismatch"),
            ErrorCode::IndexOutOfBounds => write!(f, "IndexOutOfBounds"),
            ErrorCode::EmptyConstructionRequiresRuntime => write!(f, "EmptyConstructionRequiresRuntime"),
            ErrorCode::CrossDeviceInputs => write!(f, "CrossDeviceInputs"),
            ErrorCode::CrossProvenanceInputs => write!(f, "CrossProvenanceInputs"),
            ErrorCode::RuntimeBindingMismatch => write!(f, "RuntimeBindingMismatch"),
            ErrorCode::ReadinessUnavailable => write!(f, "ReadinessUnavailable"),
            ErrorCode::ExecutionBindingUnavailable => write!(f, "ExecutionBindingUnavailable"),
            ErrorCode::LeaseAlreadyActive => write!(f, "LeaseAlreadyActive"),
            ErrorCode::UnsupportedDtype => write!(f, "UnsupportedDtype"),
            ErrorCode::UnsupportedLayout => write!(f, "UnsupportedLayout"),
            ErrorCode::UnsupportedOperation => write!(f, "UnsupportedOperation"),
            ErrorCode::OutOfMemory => write!(f, "OutOfMemory"),
            ErrorCode::CudaRuntime => write!(f, "CudaRuntime"),
            ErrorCode::FatalCuda => write!(f, "FatalCuda"),
            ErrorCode::IoFailure => write!(f, "IoFailure"),
            ErrorCode::NullUnexpected => write!(f, "NullUnexpected"),
        }
    }
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorKind::InvalidInput => write!(f, "InvalidInput"),
            ErrorKind::NotFound => write!(f, "NotFound"),
            ErrorKind::Unsupported => write!(f, "Unsupported"),
            ErrorKind::ExecutionContext => write!(f, "ExecutionContext"),
            ErrorKind::ResourceExhausted => write!(f, "ResourceExhausted"),
            ErrorKind::DependencyFailure => write!(f, "DependencyFailure"),
            ErrorKind::InvariantViolation => write!(f, "InvariantViolation"),
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
    code: ErrorCode,
    message: String,
    status: ErrorStatus,
    operation: Cow<'static, str>,
    context: Vec<(&'static str, String)>,
    source: Option<BoxedSourceError>,
}

#[derive(Debug, Clone)]
struct SourceCudfSnapshot {
    kind: ErrorKind,
    code: ErrorCode,
    status: ErrorStatus,
    operation: String,
    called: Vec<String>,
}

impl Error {
    /// Create a new error with the given kind and message.
    ///
    /// Defaults: `Permanent` status, empty operation, no context, no source.
    /// This keeps all existing call sites compiling unchanged.
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            code: default_code_for_kind(kind),
            message: message.into(),
            status: default_status_for_kind(kind),
            operation: Cow::Borrowed(""),
            context: Vec::new(),
            source: None,
        }
    }

    // -- Builder methods (consuming self, for chaining) ----------------------

    /// Set the operation that was being performed when the error occurred.
    pub fn with_operation(mut self, operation: &'static str) -> Self {
        self.push_operation(Cow::Borrowed(operation));
        self
    }

    /// Append a key-value context pair.
    pub fn with_context(mut self, key: &'static str, value: impl Into<String>) -> Self {
        self.context.push((key, value.into()));
        self
    }

    /// Set the stable machine-readable sub-code for this error.
    pub fn with_code(mut self, code: ErrorCode) -> Self {
        self.code = code;
        self
    }

    /// Attach a source error.
    pub fn set_source(mut self, src: impl Into<BoxedSourceError>) -> Self {
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

    fn push_operation(&mut self, operation: Cow<'static, str>) {
        if !self.operation.is_empty() {
            Self::push_unique_context(&mut self.context, "called", self.operation.to_string());
        }
        self.operation = operation;
    }

    fn apply_interop_context(&mut self, source_cudf_error: Option<&SourceCudfSnapshot>) {
        let mut operations = Vec::new();
        let mut remaining_context = Vec::new();

        if let Some(source) = self.source.as_deref().and_then(|source| source.downcast_ref::<InteropError>()) {
            for (key, value) in source.context() {
                if *key == "operation" {
                    operations.push(value.clone());
                } else {
                    remaining_context.push((canonical_interop_context_key(*key), value.clone()));
                }
            }
        }

        if let Some(operation) = operations.first() {
            self.operation = Cow::Owned(operation.clone());
        } else if let Some(source) = source_cudf_error.filter(|source| !source.operation.is_empty()) {
            self.operation = Cow::Owned(source.operation.clone());
        }

        for operation in operations.into_iter().skip(1) {
            Self::push_unique_context(&mut self.context, "called", operation);
        }

        if let Some(source) = source_cudf_error {
            if !source.operation.is_empty() && source.operation != self.operation() {
                Self::push_unique_context(&mut self.context, "called", source.operation.clone());
            }

            for called in &source.called {
                Self::push_unique_context(&mut self.context, "called", called.clone());
            }
        }

        for (key, value) in remaining_context {
            Self::push_unique_context(&mut self.context, key, value);
        }
    }

    fn push_unique_context(context: &mut Vec<(&'static str, String)>, key: &'static str, value: String) {
        if !context.iter().any(|(existing_key, existing_value)| *existing_key == key && *existing_value == value) {
            context.push((key, value));
        }
    }

    // -- Accessors -----------------------------------------------------------

    /// Returns the kind of this error.
    pub fn kind(&self) -> ErrorKind {
        kind_for_code(self.code).unwrap_or(self.kind)
    }

    /// Returns the stable machine-readable sub-code.
    pub fn code(&self) -> ErrorCode {
        self.code
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
    pub fn operation(&self) -> &str {
        self.operation.as_ref()
    }

    /// Returns the context key-value pairs.
    pub fn context(&self) -> &[(&'static str, String)] {
        &self.context
    }

    /// Returns the first context value for `key`, if present.
    pub fn context_value(&self, key: &str) -> Option<&str> {
        self.context.iter().find_map(|(ctx_key, ctx_value)| (*ctx_key == key).then_some(ctx_value.as_str()))
    }

    /// Returns `true` when retrying may succeed.
    pub fn is_retryable(&self) -> bool {
        self.status == ErrorStatus::Temporary
    }

    /// Returns `true` for out-of-memory failures.
    pub fn is_oom(&self) -> bool {
        self.kind() == ErrorKind::ResourceExhausted
            || matches!(self.code, ErrorCode::ResourceExhausted | ErrorCode::OutOfMemory)
    }

    /// Returns `true` for not-found failures.
    pub fn is_not_found(&self) -> bool {
        self.kind() == ErrorKind::NotFound
            || matches!(self.code, ErrorCode::NotFound | ErrorCode::MissingResource | ErrorCode::MissingColumn)
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
        write!(f, "{} ({})", self.kind(), self.status)?;

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
        writeln!(f, "{} ({}):", self.kind(), self.status)?;

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
        self.source.as_deref().map(|source| source as &(dyn StdError + 'static))
    }
}

/// Convenience type alias.
pub type Result<T> = std::result::Result<T, Error>;

pub(crate) fn invariant_violation(operation: &'static str, message: impl Into<String>) -> Error {
    Error::new(ErrorKind::InvariantViolation, message)
        .with_code(ErrorCode::InvariantViolation)
        .with_operation(operation)
}

pub(crate) fn interior_null_byte_error(
    operation: &'static str,
    resource: &'static str,
    message: &'static str,
) -> Error {
    Error::new(ErrorKind::InvalidInput, message)
        .with_code(ErrorCode::InvalidInput)
        .with_operation(operation)
        .with_context("resource", resource)
        .with_context("reason", "interior_null_byte")
}

pub(crate) fn limit_validation_error(
    kind: ErrorKind,
    operation: &'static str,
    message: impl Into<String>,
    resource: &'static str,
    value_key: &'static str,
    limit: &'static str,
    value: usize,
    remedy: impl Into<String>,
) -> Error {
    let reason = match limit {
        "i32::MAX" => "exceeds_i32_max",
        "i64::MAX" => "exceeds_i64_max",
        _ => "exceeds_supported_range",
    };

    Error::new(kind, message)
        .with_code(ErrorCode::UnsupportedOperation)
        .with_operation(operation)
        .with_context(value_key, value.to_string())
        .with_context("resource", resource)
        .with_context("reason", reason)
        .with_context("remedy", remedy.into())
}

pub(crate) fn range_limit_error(
    kind: ErrorKind,
    operation: &'static str,
    field: &'static str,
    limit: &'static str,
    value: usize,
) -> Error {
    limit_validation_error(
        kind,
        operation,
        format!("{field} exceeds {limit}"),
        field,
        field,
        limit,
        value,
        format!("reduce {field} to fit within {limit}"),
    )
}

pub(crate) fn row_range_limit_error(operation: &'static str, bound: &'static str, value: usize) -> Error {
    let (message, key, reason) = match bound {
        "start" => ("row range start exceeds i32::MAX", "start_row", "start_exceeds_i32_max"),
        "end" => ("row range end exceeds i32::MAX", "end_row", "end_exceeds_i32_max"),
        _ => ("row range bound exceeds i32::MAX", "row_bound", "bound_exceeds_i32_max"),
    };

    Error::new(ErrorKind::InvalidInput, message)
        .with_code(ErrorCode::UnsupportedOperation)
        .with_operation(operation)
        .with_context(key, value.to_string())
        .with_context("resource", "row_range")
        .with_context("reason", reason)
        .with_context("remedy", "reduce row range bounds to fit within i32::MAX")
}

pub(crate) fn row_range_overflow_error(operation: &'static str, start: usize, len: usize) -> Error {
    let end_row = (start as u128) + (len as u128);

    Error::new(ErrorKind::InvalidInput, "row range end overflowed usize")
        .with_code(ErrorCode::UnsupportedOperation)
        .with_operation(operation)
        .with_context("start_row", start.to_string())
        .with_context("end_row", end_row.to_string())
        .with_context("resource", "row_range")
        .with_context("reason", "end_overflowed_usize")
        .with_context("remedy", "reduce row range bounds to fit within usize")
}

pub(crate) fn require_non_null<T>(ptr: *mut T, operation: &'static str, target: &'static str) -> Result<NonNull<T>> {
    NonNull::new(ptr).ok_or_else(|| {
        Error::new(ErrorKind::InvariantViolation, format!("FFI returned null {target} handle on success"))
            .with_code(ErrorCode::InvariantViolation)
            .with_operation(operation)
            .with_context("target", target)
    })
}

pub(crate) fn usize_to_i32(value: usize, operation: &'static str, field: &'static str) -> Result<i32> {
    i32::try_from(value).map_err(|_| range_limit_error(ErrorKind::InvalidInput, operation, field, "i32::MAX", value))
}

pub(crate) fn usize_to_i64(value: usize, operation: &'static str, field: &'static str) -> Result<i64> {
    i64::try_from(value).map_err(|_| range_limit_error(ErrorKind::InvalidInput, operation, field, "i64::MAX", value))
}

/// Classify an [`ErrorKind`] as temporary or permanent.
///
/// CUDA runtime errors and OOM are considered temporary (the GPU may recover),
/// while logic errors (type mismatch, invalid args) are permanent.
fn default_status_for_kind(kind: ErrorKind) -> ErrorStatus {
    match kind {
        ErrorKind::ResourceExhausted => ErrorStatus::Temporary,
        _ => ErrorStatus::Permanent,
    }
}

fn default_code_for_kind(kind: ErrorKind) -> ErrorCode {
    match kind {
        ErrorKind::InvalidInput => ErrorCode::InvalidInput,
        ErrorKind::NotFound => ErrorCode::NotFound,
        ErrorKind::Unsupported => ErrorCode::Unsupported,
        ErrorKind::ExecutionContext => ErrorCode::ExecutionContext,
        ErrorKind::ResourceExhausted => ErrorCode::ResourceExhausted,
        ErrorKind::DependencyFailure => ErrorCode::DependencyFailure,
        ErrorKind::InvariantViolation => ErrorCode::InvariantViolation,
    }
}

fn map_interop_error_code(kind: InteropErrorKind) -> ErrorCode {
    match kind {
        InteropErrorKind::NotFound => ErrorCode::MissingResource,
        InteropErrorKind::UnsupportedType => ErrorCode::UnsupportedDtype,
        InteropErrorKind::NullMaskUnsupported => ErrorCode::InvalidInput,
        InteropErrorKind::InvalidView => ErrorCode::InvalidInput,
        InteropErrorKind::ExecutionBindingUnavailable => ErrorCode::ExecutionBindingUnavailable,
        InteropErrorKind::ReadinessUnavailable => ErrorCode::ReadinessUnavailable,
        InteropErrorKind::UnsupportedLayout => ErrorCode::UnsupportedLayout,
        InteropErrorKind::PlacementUnsupported | InteropErrorKind::ProfileUnsupported => {
            ErrorCode::UnsupportedOperation
        },
        InteropErrorKind::MixedDeviceUnsupported => ErrorCode::CrossDeviceInputs,
        InteropErrorKind::ExecutionBindingMismatch => ErrorCode::RuntimeBindingMismatch,
        InteropErrorKind::ReadinessMismatch => ErrorCode::RuntimeBindingMismatch,
        InteropErrorKind::LeaseAlreadyActive => ErrorCode::LeaseAlreadyActive,
    }
}

fn kind_for_code(code: ErrorCode) -> Option<ErrorKind> {
    match code {
        ErrorCode::InvalidInput
        | ErrorCode::DuplicateColumnName
        | ErrorCode::DuplicateKey
        | ErrorCode::ColumnLengthMismatch
        | ErrorCode::TypeMismatch
        | ErrorCode::IndexOutOfBounds
        | ErrorCode::NullUnexpected => Some(ErrorKind::InvalidInput),
        ErrorCode::NotFound | ErrorCode::MissingResource | ErrorCode::MissingColumn => Some(ErrorKind::NotFound),
        ErrorCode::Unsupported
        | ErrorCode::UnsupportedDtype
        | ErrorCode::UnsupportedLayout
        | ErrorCode::UnsupportedOperation => Some(ErrorKind::Unsupported),
        ErrorCode::ExecutionContext
        | ErrorCode::EmptyConstructionRequiresRuntime
        | ErrorCode::CrossDeviceInputs
        | ErrorCode::CrossProvenanceInputs
        | ErrorCode::RuntimeBindingMismatch
        | ErrorCode::ReadinessUnavailable
        | ErrorCode::ExecutionBindingUnavailable
        | ErrorCode::LeaseAlreadyActive => Some(ErrorKind::ExecutionContext),
        ErrorCode::ResourceExhausted | ErrorCode::OutOfMemory => Some(ErrorKind::ResourceExhausted),
        ErrorCode::DependencyFailure | ErrorCode::CudaRuntime | ErrorCode::FatalCuda | ErrorCode::IoFailure => {
            Some(ErrorKind::DependencyFailure)
        },
        ErrorCode::InvariantViolation => Some(ErrorKind::InvariantViolation),
    }
}

fn default_status_for_code(code: ErrorCode) -> ErrorStatus {
    match code {
        ErrorCode::ResourceExhausted | ErrorCode::OutOfMemory | ErrorCode::CudaRuntime => ErrorStatus::Temporary,
        ErrorCode::FatalCuda => ErrorStatus::Persistent,
        _ => ErrorStatus::Permanent,
    }
}

fn canonical_interop_context_key(key: &'static str) -> &'static str {
    match key {
        "producer_device" | "expected_device" | "descriptor_device" | "release_device" => "left_device",
        "runtime_device" | "actual_device" | "event_device" | "stream_device" | "consumer_device" => "right_device",
        _ => key,
    }
}

impl From<InteropError> for Error {
    fn from(error: InteropError) -> Self {
        let source_cudf_error =
            find_cudf_error_in_chain(&error as &(dyn std::error::Error + 'static)).map(|inner| SourceCudfSnapshot {
                kind: inner.kind(),
                code: inner.code(),
                status: inner.status(),
                operation: inner.operation().to_string(),
                called: inner
                    .context()
                    .iter()
                    .filter(|(key, _)| *key == "called")
                    .map(|(_, value)| value.clone())
                    .collect(),
            });
        let code = source_cudf_error.as_ref().map_or_else(|| map_interop_error_code(error.kind()), |inner| inner.code);
        let kind = source_cudf_error
            .as_ref()
            .map_or_else(|| kind_for_code(code).unwrap_or(ErrorKind::InvariantViolation), |inner| inner.kind);
        let message = error.to_string();
        let mut mapped = Error::new(kind, message).with_code(code).set_source(error);
        mapped.status = source_cudf_error.as_ref().map_or_else(|| default_status_for_code(code), |inner| inner.status);
        mapped.apply_interop_context(source_cudf_error.as_ref());
        mapped
    }
}

fn find_cudf_error_in_chain<'a>(error: &'a (dyn std::error::Error + 'static)) -> Option<&'a Error> {
    let mut current = Some(error);
    while let Some(err) = current {
        if let Some(cudf_error) = err.downcast_ref::<Error>() {
            return Some(cudf_error);
        }
        current = err.source();
    }
    None
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

    let (kind, code) = match status {
        cudf_sys::CUDF_ERROR_INVALID_ARGUMENT => (ErrorKind::InvalidInput, ErrorCode::InvalidInput),
        cudf_sys::CUDF_ERROR_INVALID_OPERATION => (ErrorKind::InvalidInput, ErrorCode::InvalidInput),
        cudf_sys::CUDF_ERROR_CUDA => (ErrorKind::DependencyFailure, ErrorCode::CudaRuntime),
        cudf_sys::CUDF_ERROR_FATAL_CUDA => (ErrorKind::DependencyFailure, ErrorCode::FatalCuda),
        cudf_sys::CUDF_ERROR_OUT_OF_MEMORY => (ErrorKind::ResourceExhausted, ErrorCode::OutOfMemory),
        cudf_sys::CUDF_ERROR_IO => (ErrorKind::DependencyFailure, ErrorCode::IoFailure),
        cudf_sys::CUDF_ERROR_TYPE => (ErrorKind::InvalidInput, ErrorCode::InvalidInput),
        cudf_sys::CUDF_ERROR_NULL => (ErrorKind::InvalidInput, ErrorCode::NullUnexpected),
        cudf_sys::CUDF_ERROR_INDEX => (ErrorKind::InvalidInput, ErrorCode::IndexOutOfBounds),
        _ => (ErrorKind::InvariantViolation, ErrorCode::InvariantViolation),
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

    let mut err = Error::new(kind, message).with_code(code);
    err.status = default_status_for_code(code);
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
    fn default_status_resource_exhausted_is_temporary() {
        assert_eq!(default_status_for_kind(ErrorKind::ResourceExhausted), ErrorStatus::Temporary);
        assert_eq!(default_status_for_code(ErrorCode::OutOfMemory), ErrorStatus::Temporary);
    }

    #[test]
    fn default_status_fatal_cuda_code_is_persistent() {
        assert_eq!(default_status_for_code(ErrorCode::FatalCuda), ErrorStatus::Persistent);
    }
}
