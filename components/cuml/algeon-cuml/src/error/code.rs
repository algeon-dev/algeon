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
    use ErrorCode::*;
    match code {
        EmptyInput
        | ShapeMismatch
        | ParameterOutOfRange
        | LabelOutOfRange
        | NonFiniteInput
        | NullUnexpected
        | ForeignInvalidArgument => ErrorKind::InvalidInput,
        UnsupportedDtype | UnsupportedLayout | UnsupportedParameter | ForeignUnsupported => ErrorKind::Unsupported,
        RuntimeBindingMismatch
        | MemoryResourceUnsupported
        | ResourcesClosed
        | ModelClosed
        | ReservationCoverageMissing
        | ForeignInvalidState => ErrorKind::ExecutionContext,
        SizeOverflow | OutOfMemory | HostOutOfMemory => ErrorKind::ResourceExhausted,
        CudaRuntime | FatalCuda | ForeignUnknown | NativeReleaseFailed => ErrorKind::DependencyFailure,
        OutputNotFinite | OutputOutOfRange | InvariantViolation => ErrorKind::InvariantViolation,
    }
}

pub const fn status_for_code(code: ErrorCode) -> ErrorStatus {
    match code {
        ErrorCode::OutOfMemory | ErrorCode::HostOutOfMemory | ErrorCode::CudaRuntime => ErrorStatus::Temporary,
        ErrorCode::FatalCuda => ErrorStatus::Persistent,
        _ => ErrorStatus::Permanent,
    }
}
