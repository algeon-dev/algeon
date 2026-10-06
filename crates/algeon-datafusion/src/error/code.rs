use std::fmt;

use query_planning::error::QueryErrorCode;

/// Declares [`ErrorCode`] from one list: each variant with its report label,
/// [`ErrorKind`], and [`ErrorStatus`]. Every table derives from the list, so a
/// new code cannot ship without all three.
macro_rules! error_codes {
    ($($(#[$doc:meta])* $code:ident => $label:literal, $kind:ident, $status:ident;)+) => {
        /// Stable, machine-readable adapter error codes.
        ///
        /// Codes describe the closest adapter-level cause. [`ErrorKind`] and
        /// [`ErrorStatus`] are derived from the code by [`ErrorCode::kind`]
        /// and [`ErrorCode::status`]; consumers should treat those methods as
        /// the policy contract rather than inferring behavior from variant
        /// names.
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        #[non_exhaustive]
        pub enum ErrorCode {
            $($(#[$doc])* $code,)+
        }

        impl ErrorCode {
            /// Every adapter error code, in declaration order.
            pub const ALL: &'static [Self] = &[$(Self::$code),+];

            /// Stable snake-case code label used in reports and error text.
            pub const fn as_str(self) -> &'static str {
                match self {
                    $(Self::$code => $label,)+
                }
            }

            /// Stable coarse category of this code.
            ///
            /// `kind` is the broad bucket used for diagnostics and
            /// aggregation. It is less specific than [`ErrorCode`] and stays
            /// stable for consumers that group failures without depending on
            /// every code variant.
            pub const fn kind(self) -> ErrorKind {
                match self {
                    $(Self::$code => ErrorKind::$kind,)+
                }
            }

            /// Retry/remediation status of this code.
            ///
            /// Attempt retry advice is derived only after terminal evidence is
            /// sealed. [`ErrorStatus::Persistent`] means a repeated or
            /// known-continuing dependency failure that requires escalation or
            /// intervention, while [`ErrorStatus::Permanent`] means the
            /// request, input, configuration, or code must change before the
            /// same operation can succeed.
            pub const fn status(self) -> ErrorStatus {
                match self {
                    $(Self::$code => ErrorStatus::$status,)+
                }
            }
        }
    };
}

error_codes! {
    /// Query execution was canceled by the caller.
    Cancelled => "cancelled", Cancelled, Permanent;
    /// SQL text could not be parsed.
    SqlParse => "sql_parse", InvalidInput, Permanent;
    /// The server request envelope or handle was invalid for the operation.
    InvalidRequest => "invalid_request", InvalidInput, Permanent;
    /// A server, backend, or device configuration value is invalid or
    /// inconsistent with the discovered hardware.
    ///
    /// The same configuration fails again until it changes; the message names
    /// the setting, its value, and the largest accepted value when one exists.
    InvalidConfiguration => "invalid_configuration", InvalidInput, Permanent;
    /// DataFusion rejected the query during logical or physical planning.
    DataFusionPlanning => "datafusion_planning", Planning, Permanent;
    /// Adapter-owned planning exceeded its configured time budget.
    ///
    /// This is classified as permanent for the same request and budget: retry
    /// with table maintenance, a narrower request, or a larger configured
    /// planning budget.
    PlanningTimeout => "planning_timeout", Planning, Permanent;
    /// A requested catalog, schema, table, table function, or similar named object was not found.
    ObjectNotFound => "object_not_found", NotFound, Permanent;
    /// Source object storage denied access to the requested object or metadata.
    SourceAccessDenied => "source_access_denied", PermissionDenied, Permanent;
    /// Source object storage lacked valid authentication credentials.
    SourceUnauthenticated => "source_unauthenticated", Unauthenticated, Permanent;
    /// Source object storage could not find the requested object.
    SourceObjectNotFound => "source_object_not_found", NotFound, Permanent;
    /// Source object storage failed with a retryable transport or service error.
    SourceTemporaryFailure => "source_temporary_failure", DependencyFailure, Temporary;
    /// A source reader or source dependency failed permanently while serving the request.
    SourceDependencyFailure => "source_dependency_failure", DependencyFailure, Permanent;
    /// A named field was not present in a resolved schema.
    FieldNotFound => "field_not_found", InvalidInput, Permanent;
    /// Materialized graph edges or side inputs violate the graph operation's
    /// declared input contract.
    GraphInputInvalid => "graph_input_invalid", InvalidInput, Permanent;
    /// A completed-plan requirement needed a GPU operation that the exact
    /// final physical plan could not provide.
    RequiredGpuOperationUnsupported => "required_gpu_operation_unsupported", Unsupported, Permanent;
    /// The exact completed physical plan did not satisfy a configured final
    /// plan requirement.
    FinalPlanRequirementUnsatisfied => "final_plan_requirement_unsatisfied", Planning, Permanent;
    /// A physical operator is unsupported by the GPU/native adapter path.
    UnsupportedOperator => "unsupported_operator", Unsupported, Permanent;
    /// A physical expression or expression form is unsupported.
    UnsupportedExpression => "unsupported_expression", Unsupported, Permanent;
    /// A data type, physical representation, or type combination is unsupported.
    UnsupportedType => "unsupported_type", Unsupported, Permanent;
    /// A DataFusion physical plan is invalid for the adapter boundary.
    ///
    /// Use this for malformed or inconsistent plans, not for well-formed plans
    /// that are merely outside the supported capability catalogue.
    InvalidExecutionPlan => "invalid_execution_plan", Planning, Permanent;
    /// An execution plan had an impossible child count for the adapter node.
    InvalidChildCount => "invalid_child_count", Internal, Permanent;
    /// cuDF rejected an input, including permanent `NotFound` conditions.
    CudfInvalidInput => "cudf_invalid_input", InvalidInput, Permanent;
    /// cuDF reported a permanent unsupported capability.
    CudfUnsupported => "cudf_unsupported", Unsupported, Permanent;
    /// A host materialization size computation overflowed the platform range.
    SizeOverflow => "size_overflow", ResourceExhausted, Permanent;
    /// One unsplittable Arrow host row exceeds the configured deterministic
    /// materialization ceiling.
    HostMaterializationLimitExceeded => "host_materialization_limit_exceeded", ResourceExhausted, Permanent;
    /// A bounded host-staging boundary exceeded its configured cumulative byte
    /// ceiling before GPU admission or a device-to-host copy.
    HostStagingLimitExceeded => "host_staging_limit_exceeded", ResourceExhausted, Permanent;
    /// DataFusion's host memory pool rejected a reservation.
    ///
    /// This is a temporary resource condition of the host-side pool only.
    /// Device memory is governed by the query domain allocator and reports
    /// through [`Self::QueryMemoryLimitExceeded`], [`Self::DeviceOutOfMemory`],
    /// and [`Self::DeviceAllocationFailed`].
    HostMemoryExhausted => "host_memory_exhausted", ResourceExhausted, Temporary;
    /// cuDF reported a temporary dependency/runtime failure.
    CudfTemporaryFailure => "cudf_temporary_failure", DependencyFailure, Temporary;
    /// cuDF reported a persistent dependency/runtime failure.
    ///
    /// The failure has been observed to recur or is known to continue. Stop
    /// automatic retries and escalate or wait for intervention.
    CudfPersistentFailure => "cudf_persistent_failure", DependencyFailure, Persistent;
    /// cuDF reported a permanent dependency failure.
    CudfDependencyFailure => "cudf_dependency_failure", DependencyFailure, Permanent;
    /// An iterative native algorithm reached its configured iteration bound.
    AlgorithmDidNotConverge => "algorithm_did_not_converge", DependencyFailure, Permanent;
    /// cuDF reported an invariant or execution-context violation.
    CudfInvariantViolation => "cudf_invariant_violation", InvariantViolation, Permanent;
    /// The native admission controller could not reserve requested memory.
    ///
    /// This is a temporary resource condition copied through from native
    /// execution, separate from a cuDF allocator failure.
    MemoryReservationFailed => "memory_reservation_failed", ResourceExhausted, Temporary;
    /// The native query service rejected work because admission capacity was exhausted.
    ServiceOverloaded => "service_overloaded", ResourceExhausted, Temporary;
    /// The requested query cap cannot fit on any eligible device after
    /// subtracting resident ownership.
    QueryBudgetUnavailable => "query_budget_unavailable", ResourceExhausted, Temporary;
    /// The query domain rejected an actual or exact-required allocation over
    /// the query's fixed memory cap.
    QueryMemoryLimitExceeded => "query_memory_limit_exceeded", ResourceExhausted, Permanent;
    /// A device allocation failed with confirmed physical-memory provenance.
    DeviceOutOfMemory => "device_out_of_memory", ResourceExhausted, Temporary;
    /// A device allocation failed without confirmed physical-memory provenance.
    DeviceAllocationFailed => "device_allocation_failed", ResourceExhausted, Temporary;
    /// A persistent device or CUDA runtime fault made the affected device
    /// unavailable.
    DeviceFault => "device_fault", DependencyFailure, Persistent;
    /// Cleanup or synchronization could not establish safe release of the
    /// attempt's device resources.
    CleanupFailed => "cleanup_failed", DependencyFailure, Persistent;
    /// An owner, provenance, ledger, or stream-routing invariant was violated.
    InternalInvariantViolation => "internal_invariant_violation", InvariantViolation, Permanent;
    /// One logical Arrow row cannot be represented within the configured
    /// Flight/gRPC message limit.
    ///
    /// This is permanent for the same row and transport limit: callers must
    /// project a smaller value or otherwise change the request. The server
    /// must not raise tonic's message-size limit to hide this condition.
    FlightFrameTooLarge => "flight_frame_too_large", ResourceExhausted, Permanent;
    /// The Flight SQL server transport failed outside request handling.
    ServerTransportFailure => "server_transport_failure", DependencyFailure, Permanent;
    /// An internal adapter bug or unreachable condition.
    Internal => "internal", Internal, Permanent;
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

/// Maps a native engine error code to the closest adapter code.
///
/// The mapping is lossy at the top-level code: native
/// `UnsupportedPipelineContract` becomes adapter `UnsupportedOperator` because
/// the adapter has no pipeline-contract code. The exact native code remains
/// available in the `native_code` fact attached by `Error::from_native`.
pub(crate) fn code_for_native_error(code: QueryErrorCode) -> ErrorCode {
    match code {
        QueryErrorCode::Cancelled => ErrorCode::Cancelled,
        QueryErrorCode::UnsupportedPipelineContract => ErrorCode::UnsupportedOperator,
        QueryErrorCode::UnsupportedOperator => ErrorCode::UnsupportedOperator,
        QueryErrorCode::UnsupportedExpression => ErrorCode::UnsupportedExpression,
        QueryErrorCode::UnsupportedType => ErrorCode::UnsupportedType,
        QueryErrorCode::InvalidPlan => ErrorCode::InvalidExecutionPlan,
        QueryErrorCode::InvalidAttemptCorrelation => ErrorCode::InvalidRequest,
        QueryErrorCode::InvalidConfiguration => ErrorCode::InvalidConfiguration,
        QueryErrorCode::GraphInputInvalid => ErrorCode::GraphInputInvalid,
        QueryErrorCode::CudfInvalidInput => ErrorCode::CudfInvalidInput,
        QueryErrorCode::CudfUnsupported => ErrorCode::CudfUnsupported,
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
        QueryErrorCode::QueryBudgetUnavailable => ErrorCode::QueryBudgetUnavailable,
        QueryErrorCode::QueryMemoryLimitExceeded => ErrorCode::QueryMemoryLimitExceeded,
        QueryErrorCode::DeviceOutOfMemory => ErrorCode::DeviceOutOfMemory,
        QueryErrorCode::DeviceAllocationFailed => ErrorCode::DeviceAllocationFailed,
        QueryErrorCode::DeviceFault => ErrorCode::DeviceFault,
        QueryErrorCode::CleanupFailed => ErrorCode::CleanupFailed,
        QueryErrorCode::InternalInvariantViolation => ErrorCode::InternalInvariantViolation,
        QueryErrorCode::Internal => ErrorCode::Internal,
    }
}
