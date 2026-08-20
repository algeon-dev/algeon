use std::fmt;

use nexus_query_engine::error::QueryErrorCode;

/// Stable, machine-readable adapter error codes.
///
/// Codes describe the closest adapter-level cause. [`ErrorKind`] and
/// [`ErrorStatus`] are derived from the code by [`kind_for_code`] and
/// [`status_for_code`]; consumers should treat those functions as the policy
/// contract rather than inferring behavior from variant names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ErrorCode {
    /// Query execution was canceled by the caller.
    Cancelled,
    /// SQL text could not be parsed.
    SqlParse,
    /// The server request envelope or handle was invalid for the operation.
    InvalidRequest,
    /// DataFusion rejected the query during logical or physical planning.
    DataFusionPlanning,
    /// Adapter-owned planning exceeded its configured time budget.
    ///
    /// This is classified as permanent for the same request and budget: retry
    /// with table maintenance, a narrower request, or a larger configured
    /// planning budget.
    PlanningTimeout,
    /// A requested catalog, schema, table, table function, or similar named object was not found.
    ObjectNotFound,
    /// Source object storage denied access to the requested object or metadata.
    SourceAccessDenied,
    /// Source object storage lacked valid authentication credentials.
    SourceUnauthenticated,
    /// Source object storage could not find the requested object.
    SourceObjectNotFound,
    /// Source object storage failed with a retryable transport or service error.
    SourceTemporaryFailure,
    /// A source reader or source dependency failed permanently while serving the request.
    SourceDependencyFailure,
    /// A named field was not present in a resolved schema.
    FieldNotFound,
    /// Materialized graph edges or side inputs violate the graph operation's
    /// declared input contract.
    GraphInputInvalid,
    /// A completed-plan requirement needed a GPU operation that the exact
    /// final physical plan could not provide.
    RequiredGpuOperationUnsupported,
    /// The exact completed physical plan did not satisfy a configured final
    /// plan requirement.
    FinalPlanRequirementUnsatisfied,
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
    /// A host materialization size computation overflowed the platform range.
    SizeOverflow,
    /// One unsplittable Arrow host row exceeds the configured deterministic
    /// materialization ceiling.
    HostMaterializationLimitExceeded,
    /// A bounded host-staging boundary exceeded its configured cumulative byte
    /// ceiling before GPU admission or a device-to-host copy.
    HostStagingLimitExceeded,
    /// cuDF reported a temporary dependency/runtime failure.
    CudfTemporaryFailure,
    /// cuDF reported a persistent dependency/runtime failure.
    ///
    /// The failure has been observed to recur or is known to continue. Stop
    /// automatic retries and escalate or wait for intervention.
    CudfPersistentFailure,
    /// cuDF reported a permanent dependency failure.
    CudfDependencyFailure,
    /// An iterative native algorithm reached its configured iteration bound.
    AlgorithmDidNotConverge,
    /// cuDF reported an invariant or execution-context violation.
    CudfInvariantViolation,
    /// The native admission controller could not reserve requested memory.
    ///
    /// This is a temporary resource condition copied through from native
    /// execution, separate from a cuDF allocator failure.
    MemoryReservationFailed,
    /// The native query service rejected work because admission capacity was exhausted.
    ServiceOverloaded,
    /// One logical Arrow row cannot be represented within the configured
    /// Flight/gRPC message limit.
    ///
    /// This is permanent for the same row and transport limit: callers must
    /// project a smaller value or otherwise change the request. The server
    /// must not raise tonic's message-size limit to hide this condition.
    FlightFrameTooLarge,
    /// The Flight SQL server transport failed outside request handling.
    ServerTransportFailure,
    /// An internal adapter bug or unreachable condition.
    Internal,
}

impl ErrorCode {
    /// Stable snake-case code label used in reports and error text.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cancelled => "cancelled",
            Self::SqlParse => "sql_parse",
            Self::InvalidRequest => "invalid_request",
            Self::DataFusionPlanning => "datafusion_planning",
            Self::PlanningTimeout => "planning_timeout",
            Self::ObjectNotFound => "object_not_found",
            Self::SourceAccessDenied => "source_access_denied",
            Self::SourceUnauthenticated => "source_unauthenticated",
            Self::SourceObjectNotFound => "source_object_not_found",
            Self::SourceTemporaryFailure => "source_temporary_failure",
            Self::SourceDependencyFailure => "source_dependency_failure",
            Self::FieldNotFound => "field_not_found",
            Self::GraphInputInvalid => "graph_input_invalid",
            Self::RequiredGpuOperationUnsupported => "required_gpu_operation_unsupported",
            Self::FinalPlanRequirementUnsatisfied => "final_plan_requirement_unsatisfied",
            Self::UnsupportedOperator => "unsupported_operator",
            Self::UnsupportedExpression => "unsupported_expression",
            Self::UnsupportedType => "unsupported_type",
            Self::InvalidExecutionPlan => "invalid_execution_plan",
            Self::InvalidChildCount => "invalid_child_count",
            Self::CudfInvalidInput => "cudf_invalid_input",
            Self::CudfUnsupported => "cudf_unsupported",
            Self::CudfResourceExhausted => "cudf_resource_exhausted",
            Self::SizeOverflow => "size_overflow",
            Self::HostMaterializationLimitExceeded => "host_materialization_limit_exceeded",
            Self::HostStagingLimitExceeded => "host_staging_limit_exceeded",
            Self::CudfTemporaryFailure => "cudf_temporary_failure",
            Self::CudfPersistentFailure => "cudf_persistent_failure",
            Self::CudfDependencyFailure => "cudf_dependency_failure",
            Self::AlgorithmDidNotConverge => "algorithm_did_not_converge",
            Self::CudfInvariantViolation => "cudf_invariant_violation",
            Self::MemoryReservationFailed => "memory_reservation_failed",
            Self::ServiceOverloaded => "service_overloaded",
            Self::FlightFrameTooLarge => "flight_frame_too_large",
            Self::ServerTransportFailure => "server_transport_failure",
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
    /// Caller-requested cancellation stopped execution.
    Cancelled,
    /// Requested catalog, schema, table, table function, or similar named object was not found.
    NotFound,
    /// The caller or configured source credentials do not have permission.
    PermissionDenied,
    /// The caller or configured source credentials are not authenticated.
    Unauthenticated,
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
            Self::Cancelled => "cancelled",
            Self::NotFound => "not_found",
            Self::PermissionDenied => "permission_denied",
            Self::Unauthenticated => "unauthenticated",
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
///
/// This is a coarse source classification, not attempt retry advice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorStatus {
    /// The same request will not succeed without changing its query, input,
    /// configuration, or code. Do not retry it automatically.
    Permanent,
    /// The failure may clear without external repair. A sealed attempt report
    /// still needs terminal evidence before it can advise backoff or retry.
    Temporary,
    /// A dependency/runtime failure has been observed to recur or is known to
    /// continue. Stop automatic retries and escalate or wait for intervention.
    ///
    /// This remains distinct from [`Self::Permanent`] because it records a
    /// continuing dependency failure rather than a request that must change;
    /// neither status alone is retry advice.
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
        ErrorCode::Cancelled => ErrorKind::Cancelled,
        ErrorCode::SqlParse => ErrorKind::InvalidInput,
        ErrorCode::InvalidRequest => ErrorKind::InvalidInput,
        ErrorCode::DataFusionPlanning
        | ErrorCode::PlanningTimeout
        | ErrorCode::FinalPlanRequirementUnsatisfied
        | ErrorCode::InvalidExecutionPlan => ErrorKind::Planning,
        ErrorCode::ObjectNotFound | ErrorCode::SourceObjectNotFound => ErrorKind::NotFound,
        ErrorCode::SourceAccessDenied => ErrorKind::PermissionDenied,
        ErrorCode::SourceUnauthenticated => ErrorKind::Unauthenticated,
        ErrorCode::FieldNotFound | ErrorCode::GraphInputInvalid => ErrorKind::InvalidInput,
        ErrorCode::RequiredGpuOperationUnsupported
        | ErrorCode::UnsupportedOperator
        | ErrorCode::UnsupportedExpression
        | ErrorCode::UnsupportedType => ErrorKind::Unsupported,
        ErrorCode::InvalidChildCount => ErrorKind::Internal,
        ErrorCode::CudfInvalidInput => ErrorKind::InvalidInput,
        ErrorCode::CudfUnsupported => ErrorKind::Unsupported,
        ErrorCode::CudfResourceExhausted
        | ErrorCode::SizeOverflow
        | ErrorCode::HostMaterializationLimitExceeded
        | ErrorCode::HostStagingLimitExceeded
        | ErrorCode::MemoryReservationFailed
        | ErrorCode::ServiceOverloaded
        | ErrorCode::FlightFrameTooLarge => ErrorKind::ResourceExhausted,
        ErrorCode::CudfTemporaryFailure
        | ErrorCode::CudfPersistentFailure
        | ErrorCode::CudfDependencyFailure
        | ErrorCode::AlgorithmDidNotConverge
        | ErrorCode::SourceDependencyFailure
        | ErrorCode::SourceTemporaryFailure
        | ErrorCode::ServerTransportFailure => ErrorKind::DependencyFailure,
        ErrorCode::CudfInvariantViolation => ErrorKind::InvariantViolation,
        ErrorCode::Internal => ErrorKind::Internal,
    }
}

/// Maps each adapter error code to its stable retry/remediation status.
///
/// Attempt retry advice is derived only after terminal evidence is sealed.
/// [`ErrorStatus::Persistent`] means a repeated or known-continuing dependency
/// failure that requires escalation or intervention, while
/// [`ErrorStatus::Permanent`] means the request, input, configuration, or code
/// must change before the same operation can succeed.
pub(crate) fn code_for_native_error(code: QueryErrorCode) -> ErrorCode {
    match code {
        QueryErrorCode::Cancelled => ErrorCode::Cancelled,
        QueryErrorCode::UnsupportedPlanShape => ErrorCode::UnsupportedOperator,
        QueryErrorCode::UnsupportedOperator => ErrorCode::UnsupportedOperator,
        QueryErrorCode::UnsupportedExpression => ErrorCode::UnsupportedExpression,
        QueryErrorCode::UnsupportedType => ErrorCode::UnsupportedType,
        QueryErrorCode::InvalidPlan => ErrorCode::InvalidExecutionPlan,
        QueryErrorCode::InvalidAttemptCorrelation => ErrorCode::InvalidRequest,
        QueryErrorCode::GraphInputInvalid => ErrorCode::GraphInputInvalid,
        QueryErrorCode::CudfInvalidInput => ErrorCode::CudfInvalidInput,
        QueryErrorCode::CudfUnsupported => ErrorCode::CudfUnsupported,
        QueryErrorCode::CudfResourceExhausted => ErrorCode::CudfResourceExhausted,
        QueryErrorCode::SizeOverflow => ErrorCode::SizeOverflow,
        QueryErrorCode::HostMaterializationLimitExceeded => ErrorCode::HostMaterializationLimitExceeded,
        QueryErrorCode::CudfTemporaryFailure => ErrorCode::CudfTemporaryFailure,
        QueryErrorCode::CudfPersistentFailure => ErrorCode::CudfPersistentFailure,
        QueryErrorCode::CudfDependencyFailure => ErrorCode::CudfDependencyFailure,
        QueryErrorCode::AlgorithmDidNotConverge => ErrorCode::AlgorithmDidNotConverge,
        QueryErrorCode::SourceAccessDenied => ErrorCode::SourceAccessDenied,
        QueryErrorCode::SourceUnauthenticated => ErrorCode::SourceUnauthenticated,
        QueryErrorCode::SourceObjectNotFound => ErrorCode::SourceObjectNotFound,
        QueryErrorCode::SourceTemporaryFailure => ErrorCode::SourceTemporaryFailure,
        QueryErrorCode::CudfInvariantViolation => ErrorCode::CudfInvariantViolation,
        QueryErrorCode::MemoryReservationFailed => ErrorCode::MemoryReservationFailed,
        QueryErrorCode::ServiceOverloaded => ErrorCode::ServiceOverloaded,
        QueryErrorCode::Internal => ErrorCode::Internal,
    }
}

pub const fn status_for_code(code: ErrorCode) -> ErrorStatus {
    match code {
        ErrorCode::Cancelled => ErrorStatus::Permanent,
        ErrorCode::CudfResourceExhausted
        | ErrorCode::CudfTemporaryFailure
        | ErrorCode::SourceTemporaryFailure
        | ErrorCode::MemoryReservationFailed
        | ErrorCode::ServiceOverloaded => ErrorStatus::Temporary,
        ErrorCode::CudfPersistentFailure => ErrorStatus::Persistent,
        ErrorCode::SqlParse
        | ErrorCode::InvalidRequest
        | ErrorCode::DataFusionPlanning
        | ErrorCode::PlanningTimeout
        | ErrorCode::ObjectNotFound
        | ErrorCode::SourceAccessDenied
        | ErrorCode::SourceUnauthenticated
        | ErrorCode::SourceObjectNotFound
        | ErrorCode::FieldNotFound
        | ErrorCode::GraphInputInvalid
        | ErrorCode::FlightFrameTooLarge
        | ErrorCode::RequiredGpuOperationUnsupported
        | ErrorCode::FinalPlanRequirementUnsatisfied
        | ErrorCode::UnsupportedOperator
        | ErrorCode::UnsupportedExpression
        | ErrorCode::UnsupportedType
        | ErrorCode::InvalidExecutionPlan
        | ErrorCode::InvalidChildCount
        | ErrorCode::CudfInvalidInput
        | ErrorCode::CudfUnsupported
        | ErrorCode::SizeOverflow
        | ErrorCode::HostMaterializationLimitExceeded
        | ErrorCode::HostStagingLimitExceeded
        | ErrorCode::CudfDependencyFailure
        | ErrorCode::AlgorithmDidNotConverge
        | ErrorCode::SourceDependencyFailure
        | ErrorCode::CudfInvariantViolation
        | ErrorCode::ServerTransportFailure
        | ErrorCode::Internal => ErrorStatus::Permanent,
    }
}
