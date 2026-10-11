use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ErrorCode {
    EmptyInput,
    ShapeMismatch,
    ParameterOutOfRange,
    LabelOutOfRange,
    NonFiniteInput,
    NullUnexpected,
    SizeOverflow,
    UnsupportedDtype,
    UnsupportedLayout,
    UnsupportedParameter,
    RuntimeBindingMismatch,
    MemoryResourceUnsupported,
    ResourcesClosed,
    ModelClosed,
    ReservationCoverageMissing,
    OutOfMemory,
    HostOutOfMemory,
    CudaRuntime,
    FatalCuda,
    ForeignInvalidArgument,
    ForeignInvalidState,
    ForeignUnsupported,
    ForeignUnknown,
    OutputNotFinite,
    OutputOutOfRange,
    NativeReleaseFailed,
    InvariantViolation,
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorKind {
    InvalidInput,
    Unsupported,
    ExecutionContext,
    ResourceExhausted,
    DependencyFailure,
    InvariantViolation,
}
impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
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
        f.write_str(match self {
            Self::Permanent => "permanent",
            Self::Temporary => "temporary",
            Self::Persistent => "persistent",
        })
    }
}

pub const fn kind_for_code(code: ErrorCode) -> ErrorKind {
    match code {
        ErrorCode::EmptyInput
        | ErrorCode::ShapeMismatch
        | ErrorCode::ParameterOutOfRange
        | ErrorCode::LabelOutOfRange
        | ErrorCode::NonFiniteInput
        | ErrorCode::NullUnexpected
        | ErrorCode::ForeignInvalidArgument => ErrorKind::InvalidInput,
        ErrorCode::UnsupportedDtype
        | ErrorCode::UnsupportedLayout
        | ErrorCode::UnsupportedParameter
        | ErrorCode::ForeignUnsupported => ErrorKind::Unsupported,
        ErrorCode::RuntimeBindingMismatch
        | ErrorCode::MemoryResourceUnsupported
        | ErrorCode::ResourcesClosed
        | ErrorCode::ModelClosed
        | ErrorCode::ReservationCoverageMissing
        | ErrorCode::ForeignInvalidState => ErrorKind::ExecutionContext,
        ErrorCode::SizeOverflow | ErrorCode::OutOfMemory | ErrorCode::HostOutOfMemory => ErrorKind::ResourceExhausted,
        ErrorCode::CudaRuntime | ErrorCode::FatalCuda | ErrorCode::ForeignUnknown | ErrorCode::NativeReleaseFailed => {
            ErrorKind::DependencyFailure
        },
        ErrorCode::OutputNotFinite | ErrorCode::OutputOutOfRange | ErrorCode::InvariantViolation => {
            ErrorKind::InvariantViolation
        },
    }
}

pub const fn status_for_code(code: ErrorCode) -> ErrorStatus {
    match code {
        ErrorCode::OutOfMemory | ErrorCode::HostOutOfMemory | ErrorCode::CudaRuntime => ErrorStatus::Temporary,
        ErrorCode::FatalCuda => ErrorStatus::Persistent,
        _ => ErrorStatus::Permanent,
    }
}
