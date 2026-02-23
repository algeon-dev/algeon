use std::fmt;

/// Stable, machine-readable adapter error codes.
///
/// Codes describe the closest adapter-level cause. [`ErrorKind`] and
/// [`ErrorStatus`] are derived from the code by [`kind_for_code`] and
/// [`status_for_code`]; consumers should treat those functions as the policy
/// contract rather than inferring behavior from variant names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ErrorCode {
    /// The optimizer intentionally rejected a native candidate.
    NativeCandidateRejected,
    /// A physical operator is unsupported by the GPU/native adapter path.
    UnsupportedOperator,
    /// A physical expression or expression form is unsupported.
    UnsupportedExpression,
    /// A data type, physical representation, or type combination is unsupported.
    UnsupportedType,
    /// A DataFusion physical plan is invalid for the adapter boundary.
    ///
    /// Use this for malformed or inconsistent plans, not for well-formed plans
    /// that are merely outside the supported capability catalogue.
    InvalidExecutionPlan,
    /// An execution plan had an impossible child count for the adapter node.
    InvalidChildCount,
    /// cuDF rejected an input, including permanent `NotFound` conditions.
    CudfInvalidInput,
    /// cuDF reported a permanent unsupported capability.
    CudfUnsupported,
    /// cuDF reported resource exhaustion.
    ///
    /// This is classified as temporary because reducing memory pressure or
    /// retrying with a different admission decision can remediate it.
    CudfResourceExhausted,
    /// cuDF reported a temporary dependency/runtime failure.
    CudfTemporaryFailure,
    /// cuDF reported a persistent dependency/runtime failure.
    ///
    /// `Persistent` is cuDF's retry classification: not immediately transient,
    /// but distinct from this crate's `Permanent` status for caller-fixable or
    /// deterministic non-retryable failures.
    CudfPersistentFailure,
    /// cuDF reported a permanent dependency failure.
    CudfDependencyFailure,
    /// cuDF reported an invariant or execution-context violation.
    CudfInvariantViolation,
    /// The native admission controller could not reserve requested memory.
    ///
    /// This is a temporary resource condition copied through from native
    /// execution, separate from a cuDF allocator failure.
    MemoryReservationFailed,
    /// An internal adapter bug or unreachable condition.
    Internal,
}

impl ErrorCode {
    /// Stable snake-case code label used in reports and error text.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NativeCandidateRejected => "native_candidate_rejected",
            Self::UnsupportedOperator => "unsupported_operator",
            Self::UnsupportedExpression => "unsupported_expression",
            Self::UnsupportedType => "unsupported_type",
            Self::InvalidExecutionPlan => "invalid_execution_plan",
            Self::InvalidChildCount => "invalid_child_count",
            Self::CudfInvalidInput => "cudf_invalid_input",
            Self::CudfUnsupported => "cudf_unsupported",
            Self::CudfResourceExhausted => "cudf_resource_exhausted",
            Self::CudfTemporaryFailure => "cudf_temporary_failure",
            Self::CudfPersistentFailure => "cudf_persistent_failure",
            Self::CudfDependencyFailure => "cudf_dependency_failure",
            Self::CudfInvariantViolation => "cudf_invariant_violation",
            Self::MemoryReservationFailed => "memory_reservation_failed",
            Self::Internal => "internal",
        }
    }

    /// Returns the stable coarse error category for this code.
    pub const fn kind(self) -> ErrorKind {
        kind_for_code(self)
    }

    /// Returns the retry/remediation status for this code.
    pub const fn status(self) -> ErrorStatus {
        status_for_code(self)
    }
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Coarse adapter error category.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ErrorKind {
    /// Planning, validation, or physical-plan construction failure.
    Planning,
    /// Caller-provided input is invalid for the requested operation.
    InvalidInput,
    /// Requested operator, expression, type, or native shape is unsupported.
    Unsupported,
    /// Runtime execution failure at the adapter boundary.
    Execution,
    /// Memory or another bounded resource was exhausted or could not be reserved.
    ResourceExhausted,
    /// cuDF or another dependency failed outside resource-exhaustion handling.
    DependencyFailure,
    /// An adapter or cuDF invariant was violated.
    InvariantViolation,
    /// Internal bug or unreachable condition.
    Internal,
}

impl ErrorKind {
    /// Stable snake-case category label used in reports and error text.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Planning => "planning",
            Self::InvalidInput => "invalid_input",
            Self::Unsupported => "unsupported",
            Self::Execution => "execution",
            Self::ResourceExhausted => "resource_exhausted",
            Self::DependencyFailure => "dependency_failure",
            Self::InvariantViolation => "invariant_violation",
            Self::Internal => "internal",
        }
    }
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Retry and remediation status for adapter errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorStatus {
    /// Non-retryable without changing the query, input, configuration, or code.
    Permanent,
    /// Retry or smaller resource admission may succeed without external repair.
    Temporary,
    /// cuDF classified the condition as not immediately transient.
    ///
    /// This name is intentionally distinct from `Permanent`: persistent failures
    /// are dependency/runtime classifications, while permanent failures are
    /// deterministic non-retryable outcomes at this API boundary.
    Persistent,
}

impl ErrorStatus {
    /// Stable snake-case status label used in reports and error text.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Permanent => "permanent",
            Self::Temporary => "temporary",
            Self::Persistent => "persistent",
        }
    }
}

impl fmt::Display for ErrorStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Maps each adapter error code to its stable coarse category.
///
/// `kind` is the broad bucket used for diagnostics and aggregation. It is less
/// specific than [`ErrorCode`] and should remain stable for consumers that group
/// failures without depending on every code variant.
pub const fn kind_for_code(code: ErrorCode) -> ErrorKind {
    match code {
        ErrorCode::NativeCandidateRejected | ErrorCode::InvalidExecutionPlan => ErrorKind::Planning,
        ErrorCode::UnsupportedOperator | ErrorCode::UnsupportedExpression | ErrorCode::UnsupportedType => {
            ErrorKind::Unsupported
        },
        ErrorCode::InvalidChildCount => ErrorKind::Internal,
        ErrorCode::CudfInvalidInput => ErrorKind::InvalidInput,
        ErrorCode::CudfUnsupported => ErrorKind::Unsupported,
        ErrorCode::CudfResourceExhausted | ErrorCode::MemoryReservationFailed => ErrorKind::ResourceExhausted,
        ErrorCode::CudfTemporaryFailure | ErrorCode::CudfPersistentFailure | ErrorCode::CudfDependencyFailure => {
            ErrorKind::DependencyFailure
        },
        ErrorCode::CudfInvariantViolation => ErrorKind::InvariantViolation,
        ErrorCode::Internal => ErrorKind::Internal,
    }
}

/// Maps each adapter error code to its stable retry/remediation status.
///
/// `status` answers what kind of remediation is plausible. `Permanent` means
/// retrying the same operation is not expected to help. `Persistent` preserves
/// cuDF's dependency/runtime status and is deliberately not a synonym for
/// `Permanent`.
pub const fn status_for_code(code: ErrorCode) -> ErrorStatus {
    match code {
        ErrorCode::CudfResourceExhausted | ErrorCode::CudfTemporaryFailure | ErrorCode::MemoryReservationFailed => {
            ErrorStatus::Temporary
        },
        ErrorCode::CudfPersistentFailure => ErrorStatus::Persistent,
        ErrorCode::NativeCandidateRejected
        | ErrorCode::UnsupportedOperator
        | ErrorCode::UnsupportedExpression
        | ErrorCode::UnsupportedType
        | ErrorCode::InvalidExecutionPlan
        | ErrorCode::InvalidChildCount
        | ErrorCode::CudfInvalidInput
        | ErrorCode::CudfUnsupported
        | ErrorCode::CudfDependencyFailure
        | ErrorCode::CudfInvariantViolation
        | ErrorCode::Internal => ErrorStatus::Permanent,
    }
}
