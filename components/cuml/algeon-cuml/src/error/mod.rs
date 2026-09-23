mod classification;
mod code;
mod ffi;

use std::{error::Error as StdError, fmt};

pub use classification::allocation_attribution;
pub use code::{ErrorCode, ErrorKind, ErrorStatus};
use cudf::{
    ErrorCode as CudfErrorCode,
    interop::contract::{InteropError, InteropErrorKind},
};
pub use ffi::{CumlErrorDetail, CumlErrorFailureSource};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FailurePhase {
    Preflight,
    NativeCall,
    OutputValidation,
    NativeRelease,
}

impl fmt::Display for FailurePhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Preflight => "preflight",
            Self::NativeCall => "native-call",
            Self::OutputValidation => "output-validation",
            Self::NativeRelease => "native-release",
        })
    }
}

struct ErrorInner {
    code: ErrorCode,
    phase: FailurePhase,
    operation: &'static str,
    message: String,
    details: Vec<(&'static str, String)>,
    ffi_detail: Option<CumlErrorDetail>,
    call_chain: Vec<&'static str>,
    source: Option<Box<dyn StdError + Send + Sync + 'static>>,
}

pub struct Error {
    inner: Box<ErrorInner>,
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    fn new(code: ErrorCode, phase: FailurePhase, operation: &'static str, message: impl Into<String>) -> Self {
        Self {
            inner: Box::new(ErrorInner {
                code,
                phase,
                operation,
                message: message.into(),
                details: vec![],
                ffi_detail: None,
                call_chain: vec![],
                source: None,
            }),
        }
    }
    pub fn code(&self) -> ErrorCode {
        self.inner.code
    }
    pub fn kind(&self) -> ErrorKind {
        code::kind_for_code(self.code())
    }
    pub fn status(&self) -> ErrorStatus {
        code::status_for_code(self.code())
    }
    pub fn phase(&self) -> FailurePhase {
        self.inner.phase
    }
    pub fn operation(&self) -> &'static str {
        self.inner.operation
    }
    pub fn message(&self) -> &str {
        &self.inner.message
    }
    pub fn details(&self) -> impl ExactSizeIterator<Item = (&'static str, &str)> {
        self.inner.details.iter().map(|(k, v)| (*k, v.as_str()))
    }
    pub fn detail_value(&self, key: &str) -> Option<&str> {
        self.inner.details.iter().rev().find(|(k, _)| *k == key).map(|(_, v)| v.as_str())
    }
    pub fn ffi_detail(&self) -> Option<&CumlErrorDetail> {
        self.inner.ffi_detail.as_ref()
    }
    pub fn call_chain(&self) -> &[&'static str] {
        &self.inner.call_chain
    }
    pub fn is_temporary(&self) -> bool {
        self.status() == ErrorStatus::Temporary
    }
    pub fn is_oom(&self) -> bool {
        matches!(self.code(), ErrorCode::OutOfMemory | ErrorCode::HostOutOfMemory)
    }
    pub fn source(&self) -> Option<&(dyn StdError + 'static)> {
        self.inner.source.as_deref().map(|e| e as &(dyn StdError + 'static))
    }
    pub(crate) fn with_detail(mut self, key: &'static str, value: impl ToString) -> Self {
        self.inner.details.push((key, value.to_string()));
        self
    }
    pub(crate) fn with_source<E: StdError + Send + Sync + 'static>(mut self, source: E) -> Self {
        self.inner.source = Some(Box::new(source));
        self
    }
    fn with_ffi_detail(mut self, detail: CumlErrorDetail) -> Self {
        self.inner.ffi_detail = Some(detail);
        self
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({}, {}) at {}: {}", self.kind(), self.status(), self.phase(), self.operation(), self.message())?;
        for (key, value) in &self.inner.details {
            write!(f, "; {key}: {value}")?;
        }
        if let Some(source) = &self.inner.source {
            write!(f, "; source: {source}")?;
        }
        Ok(())
    }
}

impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Error")
            .field("code", &self.code())
            .field("phase", &self.phase())
            .field("operation", &self.operation())
            .field("call_chain", &self.call_chain())
            .field("ffi_detail", &self.ffi_detail())
            .field("details", &self.inner.details)
            .field("source", &self.inner.source)
            .finish()
    }
}

impl StdError for Error {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        self.source()
    }
}

pub(crate) fn invalid_input(op: &'static str, msg: impl Into<String>) -> Error {
    Error::new(ErrorCode::ForeignInvalidArgument, FailurePhase::Preflight, op, msg)
}

pub(crate) fn error(code: ErrorCode, phase: FailurePhase, op: &'static str, msg: impl Into<String>) -> Error {
    Error::new(code, phase, op, msg)
}

pub(crate) fn from_cudf(op: &'static str, e: cudf::Error) -> Error {
    let code = match e.code() {
        CudfErrorCode::OutOfMemory | CudfErrorCode::ForeignResourceExhausted => ErrorCode::OutOfMemory,
        CudfErrorCode::UnsupportedDtype => ErrorCode::UnsupportedDtype,
        CudfErrorCode::UnsupportedLayout => ErrorCode::UnsupportedLayout,
        CudfErrorCode::NullUnexpected => ErrorCode::NullUnexpected,
        CudfErrorCode::SizeOverflow => ErrorCode::SizeOverflow,
        CudfErrorCode::CudaRuntime => ErrorCode::CudaRuntime,
        CudfErrorCode::FatalCuda => ErrorCode::FatalCuda,
        CudfErrorCode::CrossDeviceInputs
        | CudfErrorCode::RuntimeBindingMismatch
        | CudfErrorCode::ExecutionBindingUnavailable
        | CudfErrorCode::LeaseAlreadyActive => ErrorCode::RuntimeBindingMismatch,
        CudfErrorCode::UnsupportedOperation => ErrorCode::ForeignUnsupported,
        _ if e.kind() == cudf::ErrorKind::InvalidInput => ErrorCode::ForeignInvalidArgument,
        _ => ErrorCode::ForeignUnknown,
    };
    let mut out = Error::new(code, FailurePhase::Preflight, op, e.message())
        .with_detail("cudf_code", e.code())
        .with_detail("cudf_operation", e.operation());
    if let Some(detail) = e.ffi_detail() {
        out = out.with_ffi_detail(CumlErrorDetail {
            status: detail.status,
            failure_source: match detail.failure_source {
                cudf::CudfErrorFailureSource::Physical => CumlErrorFailureSource::Physical,
                cudf::CudfErrorFailureSource::ReservationLimit => CumlErrorFailureSource::ReservationLimit,
                _ => CumlErrorFailureSource::Unknown,
            },
            domain_id: detail.domain_id,
            member_id: detail.member_id,
            generation: detail.generation,
            requested_bytes: detail.requested_bytes,
            alignment: detail.alignment,
            cuda_error: detail.cuda_error,
            message: detail.message.clone(),
        });
    }
    out.with_source(e)
}

pub(crate) fn from_cudf_output(op: &'static str, e: cudf::Error, invalid_code: ErrorCode) -> Error {
    let invalid = e.code() == cudf::ErrorCode::ValueOutOfRange;
    let mut out = from_cudf(op, e);
    out.inner.phase = FailurePhase::OutputValidation;
    if invalid {
        out.inner.code = invalid_code;
    }
    out
}

pub(crate) fn from_interop(op: &'static str, e: InteropError) -> Error {
    let code = match e.kind() {
        InteropErrorKind::UnsupportedType => ErrorCode::UnsupportedDtype,
        InteropErrorKind::UnsupportedLayout
        | InteropErrorKind::PlacementUnsupported
        | InteropErrorKind::ProfileUnsupported => ErrorCode::UnsupportedLayout,
        InteropErrorKind::ExecutionBindingUnavailable
        | InteropErrorKind::ExecutionBindingMismatch
        | InteropErrorKind::ReadinessUnavailable
        | InteropErrorKind::ReadinessMismatch
        | InteropErrorKind::LeaseAlreadyActive => ErrorCode::RuntimeBindingMismatch,
        _ => ErrorCode::ForeignInvalidArgument,
    };
    let mut out = Error::new(code, FailurePhase::Preflight, op, e.message());
    for (key, value) in e.context() {
        out = out.with_detail(key, value);
    }
    out.with_source(e)
}

pub(crate) fn check_status(status: cuml_sys::cuml_status_t, op: &'static str) -> Result<()> {
    if status == cuml_sys::CUML_SUCCESS {
        return Ok(());
    }
    let (message, detail) = ffi::read_ffi_error(status);
    let code = match status {
        cuml_sys::CUML_ERROR_NULL | cuml_sys::CUML_ERROR_INVALID_ARGUMENT => ErrorCode::ForeignInvalidArgument,
        cuml_sys::CUML_ERROR_INVALID_STATE => ErrorCode::ForeignInvalidState,
        cuml_sys::CUML_ERROR_OUT_OF_MEMORY => ErrorCode::OutOfMemory,
        cuml_sys::CUML_ERROR_HOST_OUT_OF_MEMORY => ErrorCode::HostOutOfMemory,
        cuml_sys::CUML_ERROR_CUDA => ErrorCode::CudaRuntime,
        cuml_sys::CUML_ERROR_FATAL_CUDA => ErrorCode::FatalCuda,
        cuml_sys::CUML_ERROR_SIZE_OVERFLOW => ErrorCode::SizeOverflow,
        cuml_sys::CUML_ERROR_UNSUPPORTED => ErrorCode::ForeignUnsupported,
        _ => ErrorCode::ForeignUnknown,
    };
    let mut out = Error::new(code, FailurePhase::NativeCall, op, message);
    if let Some(detail) = detail {
        out = out.with_ffi_detail(detail);
    }
    Err(out)
}
