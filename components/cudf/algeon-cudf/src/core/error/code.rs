//! Leaf-only `ErrorCode` taxonomy and pure projections to `ErrorKind`
//! and `ErrorStatus`.
//!
//! - `ErrorCode` contains only leaf codes; it has no coarse family variants.
//! - `ErrorKind` is the coarse category, derived from `code`.
//! - `ErrorStatus` is the retry semantic, derived from `code`.
//! - Neither `ErrorKind` nor `ErrorStatus` is stored on `Error`.

use std::fmt;

/// Stable, machine-readable failure identity.
///
/// Every variant is a leaf code. Use [`ErrorKind`] if you need coarse policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ErrorCode {
    // Domestic invalid-input leaves.
    /// A DataFrame column name was expected but did not exist.
    MissingColumn,
    /// A DataFrame column name was duplicated in an input list.
    DuplicateColumnName,
    /// A lookup key was duplicated where uniqueness was required.
    DuplicateKey,
    /// Two columns expected to have matching row counts did not.
    ColumnLengthMismatch,
    /// A value's type did not match the expected type.
    TypeMismatch,
    /// A numeric, temporal, or fixed-point value could not fit in the
    /// required target range or exact representation.
    ValueOutOfRange,
    /// An integer or decimal division, or an integer remainder, had a zero
    /// divisor in a row where both operands were valid.
    DivisionByZero,
    /// A row, column, or buffer index was out of bounds.
    IndexOutOfBounds,
    /// A null was present where a non-null value was required.
    ///
    /// This leaf covers two physically distinct but contract-equivalent
    /// conditions:
    ///
    /// - An element whose mask bit is null appeared in an API that does not
    ///   accept nulls (for example, a reduction or join key column that
    ///   required non-null values).
    /// - A `&str` / `String` input contained a `\0` byte somewhere before its
    ///   end. This typically surfaces when the value must be handed to a C shim
    ///   as a null-terminated string, where the embedded `\0` would truncate the
    ///   payload.
    ///
    /// Although the two sources differ (mask bit versus string byte), downstream
    /// policy is the same. The input carries a null-ness the callee cannot
    /// process, so the caller must clean or reject it. Keeping both under one
    /// leaf avoids forcing downstreams to branch on an implementation detail
    /// that does not change their response.
    NullUnexpected,
    /// A collection-valued input was empty where at least one element was
    /// required.
    EmptyInput,
    /// A file input path pointed to a directory.
    PathIsDirectory,

    // Domestic execution-context leaves.
    /// An empty-column construction path requires an explicit runtime.
    EmptyConstructionRequiresRuntime,
    /// Inputs reside on different CUDA devices where a single-device
    /// invariant was required.
    CrossDeviceInputs,
    /// Inputs share a device but their runtime-provenance bundles conflict.
    CrossProvenanceInputs,
    /// A runtime-binding contract was violated at an execution boundary.
    RuntimeBindingMismatch,
    /// A readiness (stream/event) primitive was required but unavailable.
    ReadinessUnavailable,
    /// An execution binding (stream/memory-resource) was required but
    /// unavailable.
    ExecutionBindingUnavailable,

    // Domestic unsupported leaves.
    /// The requested dtype is not supported on this path.
    UnsupportedDtype,
    /// The requested physical layout is not supported on this path.
    UnsupportedLayout,
    /// The requested operation is not supported on this path.
    UnsupportedOperation,

    // Domestic dependency and resource leaves.
    /// Device memory (or another bounded resource) was exhausted domestically.
    OutOfMemory,
    /// A byte-size or element-count computation exceeded the representable
    /// platform range before allocation or copy.
    SizeOverflow,
    /// One unsplittable host result row exceeds a deterministic configured
    /// materialization ceiling.
    HostMaterializationLimitExceeded,
    /// A CUDA runtime call returned a potentially transient error.
    CudaRuntime,
    /// A CUDA call reported a fatal, unrecoverable error.
    FatalCuda,
    /// An I/O path (file, Arrow reader) failed.
    IoFailure,
    /// I/O cleanup could not proceed because operations are still active.
    IoBusy,

    // Domestic soft-panic leaf.
    /// An invariant in the crate's own code was violated (soft-panic replacement).
    InvariantViolation,

    // Foreign boundary leaves.
    /// The foreign layer reported caller-supplied invalid input; classification
    /// stopped at the boundary.
    ForeignInvalidArgument,
    /// The foreign layer reported a transient/runtime failure with a truthful
    /// retry signal.
    ForeignTransientDependencyFailure,
    /// The foreign layer reported a hard/intervention-required failure with a
    /// truthful non-retry signal.
    ForeignPersistentDependencyFailure,
    /// The foreign layer violated or failed to uphold its own declared
    /// contract.
    ForeignInvariantViolation,
    /// The foreign layer reported resource exhaustion with opaque
    /// classification.
    ForeignResourceExhausted,
}

impl ErrorCode {
    /// Every error code variant, in declaration order.
    pub const ALL: [Self; 33] = [
        Self::MissingColumn,
        Self::DuplicateColumnName,
        Self::DuplicateKey,
        Self::ColumnLengthMismatch,
        Self::TypeMismatch,
        Self::ValueOutOfRange,
        Self::DivisionByZero,
        Self::IndexOutOfBounds,
        Self::NullUnexpected,
        Self::EmptyInput,
        Self::PathIsDirectory,
        Self::EmptyConstructionRequiresRuntime,
        Self::CrossDeviceInputs,
        Self::CrossProvenanceInputs,
        Self::RuntimeBindingMismatch,
        Self::ReadinessUnavailable,
        Self::ExecutionBindingUnavailable,
        Self::UnsupportedDtype,
        Self::UnsupportedLayout,
        Self::UnsupportedOperation,
        Self::OutOfMemory,
        Self::SizeOverflow,
        Self::HostMaterializationLimitExceeded,
        Self::CudaRuntime,
        Self::FatalCuda,
        Self::IoFailure,
        Self::IoBusy,
        Self::InvariantViolation,
        Self::ForeignInvalidArgument,
        Self::ForeignTransientDependencyFailure,
        Self::ForeignPersistentDependencyFailure,
        Self::ForeignInvariantViolation,
        Self::ForeignResourceExhausted,
    ];
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            ErrorCode::MissingColumn => "MissingColumn",
            ErrorCode::DuplicateColumnName => "DuplicateColumnName",
            ErrorCode::DuplicateKey => "DuplicateKey",
            ErrorCode::ColumnLengthMismatch => "ColumnLengthMismatch",
            ErrorCode::TypeMismatch => "TypeMismatch",
            ErrorCode::ValueOutOfRange => "ValueOutOfRange",
            ErrorCode::DivisionByZero => "DivisionByZero",
            ErrorCode::IndexOutOfBounds => "IndexOutOfBounds",
            ErrorCode::NullUnexpected => "NullUnexpected",
            ErrorCode::EmptyConstructionRequiresRuntime => "EmptyConstructionRequiresRuntime",
            ErrorCode::EmptyInput => "EmptyInput",
            ErrorCode::PathIsDirectory => "PathIsDirectory",
            ErrorCode::CrossDeviceInputs => "CrossDeviceInputs",
            ErrorCode::CrossProvenanceInputs => "CrossProvenanceInputs",
            ErrorCode::RuntimeBindingMismatch => "RuntimeBindingMismatch",
            ErrorCode::ReadinessUnavailable => "ReadinessUnavailable",
            ErrorCode::ExecutionBindingUnavailable => "ExecutionBindingUnavailable",
            ErrorCode::UnsupportedDtype => "UnsupportedDtype",
            ErrorCode::UnsupportedLayout => "UnsupportedLayout",
            ErrorCode::UnsupportedOperation => "UnsupportedOperation",
            ErrorCode::OutOfMemory => "OutOfMemory",
            ErrorCode::SizeOverflow => "SizeOverflow",
            ErrorCode::HostMaterializationLimitExceeded => "HostMaterializationLimitExceeded",
            ErrorCode::CudaRuntime => "CudaRuntime",
            ErrorCode::FatalCuda => "FatalCuda",
            ErrorCode::IoFailure => "IoFailure",
            ErrorCode::IoBusy => "IoBusy",
            ErrorCode::InvariantViolation => "InvariantViolation",
            ErrorCode::ForeignInvalidArgument => "ForeignInvalidArgument",
            ErrorCode::ForeignTransientDependencyFailure => "ForeignTransientDependencyFailure",
            ErrorCode::ForeignPersistentDependencyFailure => "ForeignPersistentDependencyFailure",
            ErrorCode::ForeignInvariantViolation => "ForeignInvariantViolation",
            ErrorCode::ForeignResourceExhausted => "ForeignResourceExhausted",
        };
        f.write_str(name)
    }
}

/// Coarse failure category, derived from [`ErrorCode`] via [`kind_for_code`].
///
/// `ErrorKind` exists for caller policy that needs only the broad category.
/// `Error` does not store it; each call projects it from `code`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ErrorKind {
    /// The caller provided invalid values, schema requests, or foreign inputs.
    InvalidInput,
    /// A named resource referenced by the caller does not exist.
    NotFound,
    /// The requested dtype, layout, or operation is not supported.
    Unsupported,
    /// A runtime, readiness, provenance, or execution binding contract failed.
    ExecutionContext,
    /// The operation ran out of memory or another bounded resource.
    ResourceExhausted,
    /// A lower-layer CUDA, I/O, or foreign dependency failed.
    DependencyFailure,
    /// An invariant was violated (domestic or foreign).
    InvariantViolation,
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            ErrorKind::InvalidInput => "InvalidInput",
            ErrorKind::NotFound => "NotFound",
            ErrorKind::Unsupported => "Unsupported",
            ErrorKind::ExecutionContext => "ExecutionContext",
            ErrorKind::ResourceExhausted => "ResourceExhausted",
            ErrorKind::DependencyFailure => "DependencyFailure",
            ErrorKind::InvariantViolation => "InvariantViolation",
        };
        f.write_str(name)
    }
}

/// Retry semantic, derived from [`ErrorCode`] via [`status_for_code`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorStatus {
    /// Retrying with the same arguments will never succeed.
    Permanent,
    /// The failure may be transient; retrying could succeed.
    Temporary,
    /// The failure has been observed to recur; intervention is needed.
    Persistent,
}

impl fmt::Display for ErrorStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            ErrorStatus::Permanent => "permanent",
            ErrorStatus::Temporary => "temporary",
            ErrorStatus::Persistent => "persistent",
        };
        f.write_str(name)
    }
}

/// Total, pure projection from [`ErrorCode`] to [`ErrorKind`].
///
/// Every variant is mapped here; adding a new code variant without extending
/// this function is a compile error.
pub const fn kind_for_code(code: ErrorCode) -> ErrorKind {
    match code {
        ErrorCode::MissingColumn => ErrorKind::NotFound,

        ErrorCode::DuplicateColumnName
        | ErrorCode::DuplicateKey
        | ErrorCode::ColumnLengthMismatch
        | ErrorCode::TypeMismatch
        | ErrorCode::ValueOutOfRange
        | ErrorCode::DivisionByZero
        | ErrorCode::IndexOutOfBounds
        | ErrorCode::NullUnexpected
        | ErrorCode::PathIsDirectory
        | ErrorCode::ForeignInvalidArgument => ErrorKind::InvalidInput,

        ErrorCode::EmptyInput => ErrorKind::InvalidInput,

        ErrorCode::EmptyConstructionRequiresRuntime
        | ErrorCode::CrossDeviceInputs
        | ErrorCode::CrossProvenanceInputs
        | ErrorCode::RuntimeBindingMismatch
        | ErrorCode::ReadinessUnavailable
        | ErrorCode::ExecutionBindingUnavailable => ErrorKind::ExecutionContext,

        ErrorCode::UnsupportedDtype | ErrorCode::UnsupportedLayout | ErrorCode::UnsupportedOperation => {
            ErrorKind::Unsupported
        },

        ErrorCode::OutOfMemory
        | ErrorCode::SizeOverflow
        | ErrorCode::HostMaterializationLimitExceeded
        | ErrorCode::IoBusy
        | ErrorCode::ForeignResourceExhausted => ErrorKind::ResourceExhausted,

        ErrorCode::CudaRuntime
        | ErrorCode::FatalCuda
        | ErrorCode::IoFailure
        | ErrorCode::ForeignTransientDependencyFailure
        | ErrorCode::ForeignPersistentDependencyFailure => ErrorKind::DependencyFailure,

        ErrorCode::InvariantViolation | ErrorCode::ForeignInvariantViolation => ErrorKind::InvariantViolation,
    }
}

/// Total, pure projection from [`ErrorCode`] to [`ErrorStatus`].
///
/// Every variant is mapped here; adding a new code variant without extending
/// this function is a compile error.
pub const fn status_for_code(code: ErrorCode) -> ErrorStatus {
    match code {
        ErrorCode::OutOfMemory
        | ErrorCode::IoBusy
        | ErrorCode::CudaRuntime
        | ErrorCode::ForeignTransientDependencyFailure
        | ErrorCode::ForeignResourceExhausted => ErrorStatus::Temporary,

        ErrorCode::FatalCuda | ErrorCode::ForeignPersistentDependencyFailure => ErrorStatus::Persistent,

        ErrorCode::MissingColumn
        | ErrorCode::DuplicateColumnName
        | ErrorCode::DuplicateKey
        | ErrorCode::ColumnLengthMismatch
        | ErrorCode::TypeMismatch
        | ErrorCode::ValueOutOfRange
        | ErrorCode::DivisionByZero
        | ErrorCode::IndexOutOfBounds
        | ErrorCode::NullUnexpected
        | ErrorCode::EmptyConstructionRequiresRuntime
        | ErrorCode::EmptyInput
        | ErrorCode::PathIsDirectory
        | ErrorCode::CrossDeviceInputs
        | ErrorCode::CrossProvenanceInputs
        | ErrorCode::RuntimeBindingMismatch
        | ErrorCode::ReadinessUnavailable
        | ErrorCode::ExecutionBindingUnavailable
        | ErrorCode::UnsupportedDtype
        | ErrorCode::UnsupportedLayout
        | ErrorCode::UnsupportedOperation
        | ErrorCode::SizeOverflow
        | ErrorCode::HostMaterializationLimitExceeded
        | ErrorCode::IoFailure
        | ErrorCode::InvariantViolation
        | ErrorCode::ForeignInvalidArgument
        | ErrorCode::ForeignInvariantViolation => ErrorStatus::Permanent,
    }
}
