//! Error taxonomy for the DataFusion-free native query engine.
//!
//! Native errors stop at the `query-engine` API boundary as [`QueryError`].
//! The DataFusion adapter wraps them in its own `algeon_datafusion::Error` so it
//! can preserve native classification facts while exposing adapter-level
//! planning and execution categories to DataFusion.

use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error as StdError,
    fmt,
};

use serde_json::{Value, json};

use crate::{
    correlation::{QueryId, TraceId},
    triage::TriageCode,
};

type BoxedSourceError = Box<dyn StdError + Send + Sync + 'static>;

/// Native query-engine result type.
pub type Result<T> = std::result::Result<T, QueryError>;

/// Stable, machine-readable native-engine error codes.
///
/// Codes describe the closest actionable cause. [`QueryErrorKind`] and
/// [`QueryErrorStatus`] are derived from the code by [`kind_for_code`] and
/// [`status_for_code`]; consumers should not infer those policies from variant
/// names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QueryErrorCode {
    /// Query execution was cancelled by the caller.
    Cancelled,
    /// The requested IR shape is not implemented by the native executor.
    ///
    /// This is a capability rejection rather than a malformed-plan error.
    UnsupportedPlanShape,
    /// A specific operator within an otherwise recognizable plan is unsupported.
    UnsupportedOperator,
    /// An expression or expression form is unsupported.
    UnsupportedExpression,
    /// A data type, physical representation, or type combination is unsupported.
    UnsupportedType,
    /// The native IR violates executor invariants or cannot be validated.
    ///
    /// Use this when the plan is malformed for the native engine, not when the
    /// plan is well-formed but outside the supported catalogue.
    InvalidPlan,
    /// Caller-supplied execution-attempt correlation is invalid or oversized.
    InvalidAttemptCorrelation,
    /// Runtime graph inputs violate an engine-owned graph-operation contract.
    ///
    /// Unlike [`Self::InvalidPlan`], this describes materialized edge or
    /// side-input data rather than malformed native IR.
    GraphInputInvalid,
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
    /// The same request cannot be retried safely without changing its shape.
    SizeOverflow,
    /// One unsplittable Arrow host row exceeds the configured deterministic
    /// materialization ceiling.
    HostMaterializationLimitExceeded,
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
    /// Source object storage denied access to the requested object or metadata.
    SourceAccessDenied,
    /// Source object storage lacked valid authentication credentials.
    SourceUnauthenticated,
    /// Source object storage could not find the requested object.
    SourceObjectNotFound,
    /// Source object storage failed with a retryable transport or service error.
    SourceTemporaryFailure,
    /// cuDF reported an invariant or execution-context violation.
    CudfInvariantViolation,
    /// The native admission controller could not reserve requested memory.
    ///
    /// This is a temporary resource condition, separate from a cuDF allocator
    /// failure, and is intended to drive retry/backoff decisions.
    MemoryReservationFailed,
    /// The service admission layer rejected the query because shared bounded
    /// capacity is saturated.
    ServiceOverloaded,
    /// The requested query cap cannot fit on any eligible device after
    /// subtracting resident ownership.
    QueryBudgetUnavailable,
    /// The query domain rejected an actual or exact-required allocation over
    /// the query's fixed memory cap.
    QueryMemoryLimitExceeded,
    /// A device allocation failed with confirmed physical-memory provenance.
    DeviceOutOfMemory,
    /// A device allocation failed without confirmed physical-memory provenance.
    DeviceAllocationFailed,
    /// A persistent device or CUDA runtime fault made the affected device
    /// unavailable.
    DeviceFault,
    /// Cleanup or synchronization could not establish safe release of the
    /// attempt's device resources.
    CleanupFailed,
    /// An owner, provenance, ledger, or stream-routing invariant was violated.
    InternalInvariantViolation,
    /// An internal native-engine bug or unreachable condition.
    Internal,
}

impl QueryErrorCode {
    /// Every native error code supported by this engine release.
    ///
    /// This list keeps adapter projections and status policy mechanically
    /// auditable when a new code is introduced.
    pub const ALL: &'static [Self] = &[
        Self::Cancelled,
        Self::UnsupportedPlanShape,
        Self::UnsupportedOperator,
        Self::UnsupportedExpression,
        Self::UnsupportedType,
        Self::InvalidPlan,
        Self::InvalidAttemptCorrelation,
        Self::GraphInputInvalid,
        Self::CudfInvalidInput,
        Self::CudfUnsupported,
        Self::CudfResourceExhausted,
        Self::SizeOverflow,
        Self::HostMaterializationLimitExceeded,
        Self::CudfTemporaryFailure,
        Self::CudfPersistentFailure,
        Self::CudfDependencyFailure,
        Self::AlgorithmDidNotConverge,
        Self::SourceAccessDenied,
        Self::SourceUnauthenticated,
        Self::SourceObjectNotFound,
        Self::SourceTemporaryFailure,
        Self::CudfInvariantViolation,
        Self::MemoryReservationFailed,
        Self::ServiceOverloaded,
        Self::QueryBudgetUnavailable,
        Self::QueryMemoryLimitExceeded,
        Self::DeviceOutOfMemory,
        Self::DeviceAllocationFailed,
        Self::DeviceFault,
        Self::CleanupFailed,
        Self::InternalInvariantViolation,
        Self::Internal,
    ];

    /// Stable snake-case code label used in reports and error text.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cancelled => "cancelled",
            Self::UnsupportedPlanShape => "unsupported_plan_shape",
            Self::UnsupportedOperator => "unsupported_operator",
            Self::UnsupportedExpression => "unsupported_expression",
            Self::UnsupportedType => "unsupported_type",
            Self::InvalidPlan => "invalid_plan",
            Self::InvalidAttemptCorrelation => "invalid_attempt_correlation",
            Self::GraphInputInvalid => "graph_input_invalid",
            Self::CudfInvalidInput => "cudf_invalid_input",
            Self::CudfUnsupported => "cudf_unsupported",
            Self::CudfResourceExhausted => "cudf_resource_exhausted",
            Self::SizeOverflow => "size_overflow",
            Self::HostMaterializationLimitExceeded => "host_materialization_limit_exceeded",
            Self::CudfTemporaryFailure => "cudf_temporary_failure",
            Self::CudfPersistentFailure => "cudf_persistent_failure",
            Self::CudfDependencyFailure => "cudf_dependency_failure",
            Self::AlgorithmDidNotConverge => "algorithm_did_not_converge",
            Self::SourceAccessDenied => "source_access_denied",
            Self::SourceUnauthenticated => "source_unauthenticated",
            Self::SourceObjectNotFound => "source_object_not_found",
            Self::SourceTemporaryFailure => "source_temporary_failure",
            Self::CudfInvariantViolation => "cudf_invariant_violation",
            Self::MemoryReservationFailed => "memory_reservation_failed",
            Self::ServiceOverloaded => "service_overloaded",
            Self::QueryBudgetUnavailable => "query_budget_unavailable",
            Self::QueryMemoryLimitExceeded => "query_memory_limit_exceeded",
            Self::DeviceOutOfMemory => "device_out_of_memory",
            Self::DeviceAllocationFailed => "device_allocation_failed",
            Self::DeviceFault => "device_fault",
            Self::CleanupFailed => "cleanup_failed",
            Self::InternalInvariantViolation => "internal_invariant_violation",
            Self::Internal => "internal",
        }
    }

    /// Returns the stable coarse error category for this code.
    pub const fn kind(self) -> QueryErrorKind {
        kind_for_code(self)
    }

    /// Returns the retry/remediation status for this code.
    pub const fn status(self) -> QueryErrorStatus {
        status_for_code(self)
    }
}

impl fmt::Display for QueryErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Coarse native-engine error category.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum QueryErrorKind {
    /// Caller-requested cancellation stopped execution.
    Cancelled,
    /// Planning, validation, or IR construction failure.
    Planning,
    /// Caller-provided input is invalid for the requested operation.
    InvalidInput,
    /// Requested plan, operator, expression, or type is not supported.
    Unsupported,
    /// Memory or other bounded resource was exhausted or could not be reserved.
    ResourceExhausted,
    /// cuDF or another dependency failed outside resource-exhaustion handling.
    DependencyFailure,
    /// A native-engine or cuDF invariant was violated.
    InvariantViolation,
    /// Internal bug or unreachable condition.
    Internal,
}

impl QueryErrorKind {
    /// Stable snake-case category label used in reports and error text.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cancelled => "cancelled",
            Self::Planning => "planning",
            Self::InvalidInput => "invalid_input",
            Self::Unsupported => "unsupported",
            Self::ResourceExhausted => "resource_exhausted",
            Self::DependencyFailure => "dependency_failure",
            Self::InvariantViolation => "invariant_violation",
            Self::Internal => "internal",
        }
    }
}

impl fmt::Display for QueryErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Retry and remediation status for native-engine errors.
///
/// This is a coarse source classification, not attempt retry advice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QueryErrorStatus {
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

impl QueryErrorStatus {
    /// Stable snake-case status label used in reports and error text.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Permanent => "permanent",
            Self::Temporary => "temporary",
            Self::Persistent => "persistent",
        }
    }
}

impl fmt::Display for QueryErrorStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Maps each native error code to its stable coarse category.
///
/// `kind` is the broad bucket used for diagnostics and aggregation. It is less
/// specific than [`QueryErrorCode`] and should remain stable for consumers that
/// group failures without depending on every code variant.
pub const fn kind_for_code(code: QueryErrorCode) -> QueryErrorKind {
    match code {
        QueryErrorCode::Cancelled => QueryErrorKind::Cancelled,
        QueryErrorCode::UnsupportedPlanShape
        | QueryErrorCode::UnsupportedOperator
        | QueryErrorCode::UnsupportedExpression
        | QueryErrorCode::UnsupportedType => QueryErrorKind::Unsupported,
        QueryErrorCode::InvalidPlan => QueryErrorKind::Planning,
        QueryErrorCode::InvalidAttemptCorrelation => QueryErrorKind::InvalidInput,
        QueryErrorCode::GraphInputInvalid => QueryErrorKind::InvalidInput,
        QueryErrorCode::CudfInvalidInput => QueryErrorKind::InvalidInput,
        QueryErrorCode::CudfUnsupported => QueryErrorKind::Unsupported,
        QueryErrorCode::CudfResourceExhausted
        | QueryErrorCode::SizeOverflow
        | QueryErrorCode::HostMaterializationLimitExceeded
        | QueryErrorCode::MemoryReservationFailed
        | QueryErrorCode::ServiceOverloaded
        | QueryErrorCode::QueryBudgetUnavailable
        | QueryErrorCode::QueryMemoryLimitExceeded
        | QueryErrorCode::DeviceOutOfMemory
        | QueryErrorCode::DeviceAllocationFailed => QueryErrorKind::ResourceExhausted,
        QueryErrorCode::CudfTemporaryFailure
        | QueryErrorCode::CudfPersistentFailure
        | QueryErrorCode::CudfDependencyFailure
        | QueryErrorCode::AlgorithmDidNotConverge
        | QueryErrorCode::SourceAccessDenied
        | QueryErrorCode::SourceUnauthenticated
        | QueryErrorCode::SourceObjectNotFound
        | QueryErrorCode::SourceTemporaryFailure
        | QueryErrorCode::DeviceFault
        | QueryErrorCode::CleanupFailed => QueryErrorKind::DependencyFailure,
        QueryErrorCode::CudfInvariantViolation | QueryErrorCode::InternalInvariantViolation => {
            QueryErrorKind::InvariantViolation
        },
        QueryErrorCode::Internal => QueryErrorKind::Internal,
    }
}

/// Maps each native error code to its stable retry/remediation status.
///
/// Attempt retry advice is derived only after terminal evidence is sealed.
/// [`QueryErrorStatus::Persistent`] means a repeated or known-continuing
/// dependency failure that requires escalation or intervention, while
/// [`QueryErrorStatus::Permanent`] means the request, input, configuration, or
/// code must change before the same operation can succeed.
pub const fn status_for_code(code: QueryErrorCode) -> QueryErrorStatus {
    match code {
        QueryErrorCode::Cancelled => QueryErrorStatus::Permanent,
        QueryErrorCode::CudfResourceExhausted
        | QueryErrorCode::CudfTemporaryFailure
        | QueryErrorCode::SourceTemporaryFailure
        | QueryErrorCode::MemoryReservationFailed
        | QueryErrorCode::ServiceOverloaded
        | QueryErrorCode::QueryBudgetUnavailable
        | QueryErrorCode::DeviceOutOfMemory
        | QueryErrorCode::DeviceAllocationFailed => QueryErrorStatus::Temporary,
        QueryErrorCode::CudfPersistentFailure => QueryErrorStatus::Persistent,
        QueryErrorCode::UnsupportedPlanShape
        | QueryErrorCode::UnsupportedOperator
        | QueryErrorCode::UnsupportedExpression
        | QueryErrorCode::UnsupportedType
        | QueryErrorCode::InvalidPlan
        | QueryErrorCode::InvalidAttemptCorrelation
        | QueryErrorCode::GraphInputInvalid
        | QueryErrorCode::CudfInvalidInput
        | QueryErrorCode::CudfUnsupported
        | QueryErrorCode::SizeOverflow
        | QueryErrorCode::HostMaterializationLimitExceeded
        | QueryErrorCode::CudfDependencyFailure
        | QueryErrorCode::AlgorithmDidNotConverge
        | QueryErrorCode::SourceAccessDenied
        | QueryErrorCode::SourceUnauthenticated
        | QueryErrorCode::SourceObjectNotFound
        | QueryErrorCode::CudfInvariantViolation
        | QueryErrorCode::QueryMemoryLimitExceeded
        | QueryErrorCode::InternalInvariantViolation => QueryErrorStatus::Permanent,
        QueryErrorCode::DeviceFault | QueryErrorCode::CleanupFailed => QueryErrorStatus::Persistent,
        QueryErrorCode::Internal => QueryErrorStatus::Permanent,
    }
}

#[doc(hidden)]
pub const fn triage_code_for_query_error(code: QueryErrorCode) -> TriageCode {
    match code {
        QueryErrorCode::Cancelled => TriageCode::Cancelled,
        QueryErrorCode::UnsupportedPlanShape => TriageCode::UnsupportedPlanShape,
        QueryErrorCode::UnsupportedOperator => TriageCode::UnsupportedOperator,
        QueryErrorCode::UnsupportedExpression => TriageCode::UnsupportedExpression,
        QueryErrorCode::UnsupportedType => TriageCode::UnsupportedType,
        QueryErrorCode::InvalidPlan => TriageCode::InvalidPlan,
        QueryErrorCode::InvalidAttemptCorrelation
        | QueryErrorCode::GraphInputInvalid
        | QueryErrorCode::CudfInvalidInput => TriageCode::InvalidInput,
        QueryErrorCode::CudfUnsupported => TriageCode::UnsupportedPlanShape,
        QueryErrorCode::CudfResourceExhausted
        | QueryErrorCode::SizeOverflow
        | QueryErrorCode::HostMaterializationLimitExceeded
        | QueryErrorCode::QueryMemoryLimitExceeded
        | QueryErrorCode::DeviceOutOfMemory
        | QueryErrorCode::DeviceAllocationFailed => TriageCode::ResourceExhausted,
        QueryErrorCode::CudfTemporaryFailure
        | QueryErrorCode::CudfPersistentFailure
        | QueryErrorCode::CudfDependencyFailure
        | QueryErrorCode::AlgorithmDidNotConverge => TriageCode::DependencyFailure,
        QueryErrorCode::CudfInvariantViolation => TriageCode::InvariantViolation,
        QueryErrorCode::ServiceOverloaded => TriageCode::ServiceOverloaded,
        QueryErrorCode::Internal => TriageCode::Internal,
        QueryErrorCode::SourceAccessDenied
        | QueryErrorCode::SourceUnauthenticated
        | QueryErrorCode::SourceObjectNotFound
        | QueryErrorCode::SourceTemporaryFailure => TriageCode::SourceFailure,
        QueryErrorCode::MemoryReservationFailed => TriageCode::MemoryReservationDenied,
        QueryErrorCode::QueryBudgetUnavailable => TriageCode::MemoryReservationDenied,
        QueryErrorCode::DeviceFault | QueryErrorCode::CleanupFailed => TriageCode::DependencyFailure,
        QueryErrorCode::InternalInvariantViolation => TriageCode::InvariantViolation,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
#[doc(hidden)]
pub enum AttemptFailureCode {
    Cancelled,
    UnsupportedPlanShape,
    UnsupportedOperator,
    UnsupportedExpression,
    UnsupportedType,
    InvalidPlan,
    InvalidAttemptCorrelation,
    GraphInputInvalid,
    CudfInvalidInput,
    CudfUnsupported,
    CudfResourceExhausted,
    SizeOverflow,
    HostMaterializationLimitExceeded,
    CudfTemporaryFailure,
    CudfPersistentFailure,
    CudfDependencyFailure,
    AlgorithmDidNotConverge,
    SourceAccessDenied,
    SourceUnauthenticated,
    SourceObjectNotFound,
    SourceTemporaryFailure,
    CudfInvariantViolation,
    MemoryReservationFailed,
    ServiceOverloaded,
    QueryBudgetUnavailable,
    QueryMemoryLimitExceeded,
    DeviceOutOfMemory,
    DeviceAllocationFailed,
    DeviceFault,
    CleanupFailed,
    InternalInvariantViolation,
    Internal,
    BackendClosing,
    DatafusionError,
    Dropped,
    Panicked,
    StreamSynchronizationFailed,
    CleanupCallbackAbandoned,
}

impl AttemptFailureCode {
    pub const fn from_query_error_code(code: QueryErrorCode) -> Self {
        attempt_failure_code_for_query_error(code)
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cancelled => "cancelled",
            Self::UnsupportedPlanShape => "unsupported_plan_shape",
            Self::UnsupportedOperator => "unsupported_operator",
            Self::UnsupportedExpression => "unsupported_expression",
            Self::UnsupportedType => "unsupported_type",
            Self::InvalidPlan => "invalid_plan",
            Self::InvalidAttemptCorrelation => "invalid_attempt_correlation",
            Self::GraphInputInvalid => "graph_input_invalid",
            Self::CudfInvalidInput => "cudf_invalid_input",
            Self::CudfUnsupported => "cudf_unsupported",
            Self::CudfResourceExhausted => "cudf_resource_exhausted",
            Self::SizeOverflow => "size_overflow",
            Self::HostMaterializationLimitExceeded => "host_materialization_limit_exceeded",
            Self::CudfTemporaryFailure => "cudf_temporary_failure",
            Self::CudfPersistentFailure => "cudf_persistent_failure",
            Self::CudfDependencyFailure => "cudf_dependency_failure",
            Self::AlgorithmDidNotConverge => "algorithm_did_not_converge",
            Self::SourceAccessDenied => "source_access_denied",
            Self::SourceUnauthenticated => "source_unauthenticated",
            Self::SourceObjectNotFound => "source_object_not_found",
            Self::SourceTemporaryFailure => "source_temporary_failure",
            Self::CudfInvariantViolation => "cudf_invariant_violation",
            Self::MemoryReservationFailed => "memory_reservation_failed",
            Self::ServiceOverloaded => "service_overloaded",
            Self::QueryBudgetUnavailable => "query_budget_unavailable",
            Self::QueryMemoryLimitExceeded => "query_memory_limit_exceeded",
            Self::DeviceOutOfMemory => "device_out_of_memory",
            Self::DeviceAllocationFailed => "device_allocation_failed",
            Self::DeviceFault => "device_fault",
            Self::CleanupFailed => "cleanup_failed",
            Self::InternalInvariantViolation => "internal_invariant_violation",
            Self::Internal => "internal",
            Self::BackendClosing => "backend_closing",
            Self::DatafusionError => "datafusion_error",
            Self::Dropped => "dropped",
            Self::Panicked => "panicked",
            Self::StreamSynchronizationFailed => "stream_synchronization_failed",
            Self::CleanupCallbackAbandoned => "cleanup_callback_abandoned",
        }
    }
}

#[doc(hidden)]
pub const fn attempt_failure_code_for_query_error(code: QueryErrorCode) -> AttemptFailureCode {
    match code {
        QueryErrorCode::Cancelled => AttemptFailureCode::Cancelled,
        QueryErrorCode::UnsupportedPlanShape => AttemptFailureCode::UnsupportedPlanShape,
        QueryErrorCode::UnsupportedOperator => AttemptFailureCode::UnsupportedOperator,
        QueryErrorCode::UnsupportedExpression => AttemptFailureCode::UnsupportedExpression,
        QueryErrorCode::UnsupportedType => AttemptFailureCode::UnsupportedType,
        QueryErrorCode::InvalidPlan => AttemptFailureCode::InvalidPlan,
        QueryErrorCode::InvalidAttemptCorrelation => AttemptFailureCode::InvalidAttemptCorrelation,
        QueryErrorCode::GraphInputInvalid => AttemptFailureCode::GraphInputInvalid,
        QueryErrorCode::CudfInvalidInput => AttemptFailureCode::CudfInvalidInput,
        QueryErrorCode::CudfUnsupported => AttemptFailureCode::CudfUnsupported,
        QueryErrorCode::CudfResourceExhausted => AttemptFailureCode::CudfResourceExhausted,
        QueryErrorCode::SizeOverflow => AttemptFailureCode::SizeOverflow,
        QueryErrorCode::HostMaterializationLimitExceeded => AttemptFailureCode::HostMaterializationLimitExceeded,
        QueryErrorCode::CudfTemporaryFailure => AttemptFailureCode::CudfTemporaryFailure,
        QueryErrorCode::CudfPersistentFailure => AttemptFailureCode::CudfPersistentFailure,
        QueryErrorCode::CudfDependencyFailure => AttemptFailureCode::CudfDependencyFailure,
        QueryErrorCode::AlgorithmDidNotConverge => AttemptFailureCode::AlgorithmDidNotConverge,
        QueryErrorCode::SourceAccessDenied => AttemptFailureCode::SourceAccessDenied,
        QueryErrorCode::SourceUnauthenticated => AttemptFailureCode::SourceUnauthenticated,
        QueryErrorCode::SourceObjectNotFound => AttemptFailureCode::SourceObjectNotFound,
        QueryErrorCode::SourceTemporaryFailure => AttemptFailureCode::SourceTemporaryFailure,
        QueryErrorCode::CudfInvariantViolation => AttemptFailureCode::CudfInvariantViolation,
        QueryErrorCode::MemoryReservationFailed => AttemptFailureCode::MemoryReservationFailed,
        QueryErrorCode::ServiceOverloaded => AttemptFailureCode::ServiceOverloaded,
        QueryErrorCode::QueryBudgetUnavailable => AttemptFailureCode::QueryBudgetUnavailable,
        QueryErrorCode::QueryMemoryLimitExceeded => AttemptFailureCode::QueryMemoryLimitExceeded,
        QueryErrorCode::DeviceOutOfMemory => AttemptFailureCode::DeviceOutOfMemory,
        QueryErrorCode::DeviceAllocationFailed => AttemptFailureCode::DeviceAllocationFailed,
        QueryErrorCode::DeviceFault => AttemptFailureCode::DeviceFault,
        QueryErrorCode::CleanupFailed => AttemptFailureCode::CleanupFailed,
        QueryErrorCode::InternalInvariantViolation => AttemptFailureCode::InternalInvariantViolation,
        QueryErrorCode::Internal => AttemptFailureCode::Internal,
    }
}

/// Machine-readable facts attached to a [`QueryError`].
///
/// Fact keys are static strings so they can be treated as a documented contract
/// by tests, reports, and callers. Values are stringified with [`ToString`].
/// Iteration is sorted by key because facts are stored in a [`BTreeMap`].
///
/// Decoration never panics: empty keys are ignored and duplicate inserts keep
/// the first value. Rejected keys are reported deterministically under
/// `fact_collision_keys` so a colliding decoration stays diagnosable instead
/// of replacing the primary error's facts. Callers that own a key and must
/// overwrite it use [`QueryError::with_query_id`], [`QueryError::with_trace_id`],
/// or another explicit overwrite path, never a second [`QueryError::with_fact`].
/// The native engine reserves `cudf_code`, `cudf_kind`, `cudf_status`,
/// `cudf_operation`, and `cudf_message` for wrapped `cudf::Error` facts.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Facts {
    values: BTreeMap<&'static str, String>,
    collision_keys: BTreeSet<&'static str>,
}

/// Fact key recording rejected duplicate or empty decoration keys.
pub const FACT_COLLISION_KEYS: &str = "fact_collision_keys";

/// Placeholder recorded when a decoration key is empty.
pub const EMPTY_FACT_KEY: &str = "<empty>";

impl Facts {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get(&self, key: &str) -> Option<&str> {
        self.values.get(key).map(String::as_str)
    }

    /// Iterates facts in sorted key order.
    pub fn iter(&self) -> impl Iterator<Item = (&'static str, &str)> {
        self.values.iter().map(|(key, value)| (*key, value.as_str()))
    }

    fn insert(&mut self, key: &'static str, value: String) {
        if key.is_empty() {
            self.record_collision(EMPTY_FACT_KEY);
            return;
        }
        if key == FACT_COLLISION_KEYS || self.values.contains_key(key) {
            self.record_collision(key);
            return;
        }
        self.values.insert(key, value);
    }

    fn record_collision(&mut self, key: &'static str) {
        self.collision_keys.insert(key);
        let value = self.collision_keys.iter().copied().collect::<Vec<_>>().join(",");
        self.values.insert(FACT_COLLISION_KEYS, value);
    }

    fn set(&mut self, key: &'static str, value: String) {
        assert!(!key.is_empty(), "query-engine error fact key must not be empty");
        self.values.insert(key, value);
    }
}

/// Structured native-engine error with stable classification and facts.
pub struct QueryError(Box<QueryErrorRepr>);

struct QueryErrorRepr {
    query_id: Option<QueryId>,
    trace_id: Option<TraceId>,
    code: QueryErrorCode,
    operation: &'static str,
    message: String,
    facts: Box<Facts>,
    cugraph_allocation_source: Option<CugraphAllocationSource>,
    graph_cache_scope_peak_bytes: Option<u64>,
    source: Option<BoxedSourceError>,
    cleanup_disposition: Option<CleanupDisposition>,
}

/// Closed cuGraph allocation-failure provenance retained independently of
/// diagnostic string facts.
///
/// `RmmOutOfMemory` is the callback-backed path through the query memory
/// resource. The `bad_alloc` variants identify direct cuGraph/RAFT throwing
/// paths; when they traverse a resident cache allocation scope that scope
/// publishes an exact allocator event independently. `Unknown` remains
/// fail-closed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CugraphAllocationSource {
    Unknown,
    RmmOutOfMemory,
    RmmBadAlloc,
    StdBadAlloc,
}

impl CugraphAllocationSource {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::RmmOutOfMemory => "rmm_out_of_memory",
            Self::RmmBadAlloc => "rmm_bad_alloc",
            Self::StdBadAlloc => "std_bad_alloc",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CleanupDisposition {
    DeadlineElapsed,
}

impl QueryError {
    /// Creates a native-engine error with a stable operation id and message.
    ///
    /// `operation` should be a stable dotted identifier such as
    /// `query_engine.plan.execute`. `message` is human-facing diagnostic
    /// text. Add machine-readable context with [`Self::with_fact`].
    ///
    /// # Panics
    ///
    /// Panics when `operation` or `message` is empty. Construction-time
    /// identity must be present; later decoration ([`Self::with_fact`],
    /// [`Self::with_source`]) never panics and keeps the first value instead.
    pub fn new(code: QueryErrorCode, operation: &'static str, message: impl Into<String>) -> Self {
        let message = message.into();
        assert!(!operation.is_empty(), "query-engine error operation must not be empty");
        assert!(!message.is_empty(), "query-engine error message must not be empty");
        Self(Box::new(QueryErrorRepr {
            query_id: None,
            trace_id: None,
            code,
            operation,
            message,
            facts: Box::new(Facts::new()),
            cugraph_allocation_source: None,
            graph_cache_scope_peak_bytes: None,
            source: None,
            cleanup_disposition: None,
        }))
    }

    /// Marks an owner-reported cleanup wait as an elapsed deadline.
    ///
    /// This internal typed bit is deliberately separate from the public error
    /// code: `ServiceOverloaded` is also used by ordinary admission and cannot
    /// by itself authorize indefinite cleanup retries.
    #[must_use]
    #[doc(hidden)]
    pub fn with_cleanup_deadline_elapsed(mut self) -> Self {
        self.0.cleanup_disposition = Some(CleanupDisposition::DeadlineElapsed);
        self
    }

    #[doc(hidden)]
    pub fn is_cleanup_deadline_elapsed(&self) -> bool {
        self.0.cleanup_disposition == Some(CleanupDisposition::DeadlineElapsed)
    }

    /// Adds one machine-readable fact.
    ///
    /// `key` must be a non-empty static string and should be treated as part of
    /// the diagnostic contract once exposed. `value` is stringified with
    /// [`ToString`]. A duplicate or empty key never panics and never replaces
    /// the primary error's facts: the first value wins and the rejected key is
    /// recorded under [`FACT_COLLISION_KEYS`]. Layered `with_fact` calls must
    /// therefore use distinct keys per layer; wrapping an error must not
    /// re-emit a key the wrapped error already owns.
    #[must_use]
    pub fn with_fact(mut self, key: &'static str, value: impl ToString) -> Self {
        self.0.facts.insert(key, value.to_string());
        self
    }

    #[must_use]
    #[doc(hidden)]
    pub fn with_fact_if_absent(mut self, key: &'static str, value: impl ToString) -> Self {
        if self.0.facts.get(key).is_none() {
            self.0.facts.insert(key, value.to_string());
        }
        self
    }

    /// Retains typed cuGraph allocation provenance and mirrors its stable label
    /// into ordinary error facts for existing diagnostic consumers.
    ///
    /// This is an explicit overwrite path owned by the cuGraph boundary: the
    /// mirrored fact always tracks the typed field, so a second call replaces
    /// both together instead of recording a collision.
    #[cfg(any(feature = "cugraph", feature = "test-hooks", test))]
    #[must_use]
    #[doc(hidden)]
    pub fn with_cugraph_allocation_source(mut self, source: CugraphAllocationSource) -> Self {
        self.0.facts.set("cugraph_allocation_source", source.as_str().to_string());
        self.0.cugraph_allocation_source = Some(source);
        self
    }

    /// Returns the closed cuGraph allocation provenance, when the cuGraph
    /// boundary supplied one.
    pub const fn cugraph_allocation_source(&self) -> Option<CugraphAllocationSource> {
        self.0.cugraph_allocation_source
    }

    /// Retains an exact high-water mark from a completed resident-graph cache
    /// scope. This is an internal typed sideband, deliberately distinct from
    /// error facts so consumers cannot mistake diagnostic text for a peak.
    #[cfg(feature = "cugraph")]
    #[must_use]
    #[doc(hidden)]
    pub fn with_graph_cache_scope_peak_bytes(mut self, peak_bytes: u64) -> Self {
        self.0.graph_cache_scope_peak_bytes = Some(peak_bytes);
        self
    }

    /// Returns the exact resident-cache scope high-water carried from a graph
    /// execution error only after that scope completed its fence and health
    /// proof.
    #[cfg(feature = "cugraph")]
    #[doc(hidden)]
    pub const fn graph_cache_scope_peak_bytes(&self) -> Option<u64> {
        self.0.graph_cache_scope_peak_bytes
    }

    /// Assigns a query-wide diagnostic correlation id and mirrors it into facts.
    #[must_use]
    pub fn with_query_id(mut self, query_id: impl Into<QueryId>) -> Self {
        let query_id = query_id.into();
        self.0.facts.set("query_id", query_id.to_string());
        self.0.query_id = Some(query_id);
        self
    }

    /// Assigns an optional parent trace id and mirrors it into facts.
    #[must_use]
    pub fn with_trace_id(mut self, trace_id: impl Into<TraceId>) -> Self {
        let trace_id = trace_id.into();
        self.0.facts.set("trace_id", trace_id.to_string());
        self.0.trace_id = Some(trace_id);
        self
    }

    /// Attaches the source error when no source is already present.
    ///
    /// A second source is discarded so error decoration cannot replace the
    /// original causal chain. Layers that need to add context wrap the error
    /// in a new [`QueryError`] instead of re-attaching a source.
    #[must_use]
    pub fn with_source<E>(mut self, source: E) -> Self
    where E: StdError + Send + Sync + 'static {
        if self.0.source.is_none() {
            self.0.source = Some(Box::new(source));
        }
        self
    }

    /// Returns the stable machine-readable error code.
    pub fn code(&self) -> QueryErrorCode {
        self.0.code
    }

    /// Returns the coarse category derived from [`Self::code`].
    pub fn kind(&self) -> QueryErrorKind {
        self.0.code.kind()
    }

    /// Returns the retry/remediation status derived from [`Self::code`].
    pub fn status(&self) -> QueryErrorStatus {
        self.0.code.status()
    }

    /// Returns the query-wide diagnostic correlation id, when assigned.
    pub fn query_id(&self) -> Option<&QueryId> {
        self.0.query_id.as_ref()
    }

    /// Returns the optional parent trace id, when assigned.
    pub fn trace_id(&self) -> Option<&TraceId> {
        self.0.trace_id.as_ref()
    }

    /// Returns the stable dotted operation id that produced the error.
    pub fn operation(&self) -> &'static str {
        self.0.operation
    }

    /// Returns the human-facing diagnostic message.
    pub fn message(&self) -> &str {
        &self.0.message
    }

    /// Returns the sorted machine-readable fact set.
    pub fn facts(&self) -> &Facts {
        self.0.facts.as_ref()
    }

    pub fn fact_value(&self, key: &str) -> Option<&str> {
        self.0.facts.get(key)
    }

    /// Returns the underlying source error when one was attached.
    pub fn source_error(&self) -> Option<&(dyn StdError + 'static)> {
        self.0.source.as_deref().map(|source| source as &(dyn StdError + 'static))
    }

    /// Serializes this error as a structured diagnostic JSON object.
    pub fn to_json(&self) -> Value {
        let query_id = self.0.query_id.as_ref().map(QueryId::as_str).or_else(|| self.fact_value("query_id"));
        let trace_id = self.0.trace_id.as_ref().map(TraceId::as_str).or_else(|| self.fact_value("trace_id"));
        json!({
            "schema": "query_engine.query_error.v1",
            "query_id": query_id,
            "trace_id": trace_id,
            "triage": self.triage_summary().to_json(),
            "code": self.0.code.as_str(),
            "kind": self.kind().as_str(),
            "status": self.status().as_str(),
            "operation": self.0.operation,
            "message": self.0.message,
            "facts": self.0.facts.iter().collect::<BTreeMap<_, _>>(),
            "source": self.0.source.as_ref().map(|source| source.to_string()),
        })
    }
}

/// Redacts whitespace-delimited `scheme://...` tokens and normalizes spaces.
///
/// Native boundaries project foreign error text into diagnostic facts. They
/// share this helper so one redaction policy covers every such projection,
/// whichever component produced the message.
pub fn redact_uri_tokens(message: &str) -> String {
    message
        .split_whitespace()
        .map(|token| if token.contains("://") { "<redacted-uri>" } else { token })
        .collect::<Vec<_>>()
        .join(" ")
}

impl fmt::Display for QueryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "code={} kind={} status={} operation={} message={}",
            self.code(),
            self.kind(),
            self.status(),
            self.operation(),
            quote(self.message()),
        )?;

        if self.0.facts.values.is_empty() {
            f.write_str(" facts={}")?;
        } else {
            f.write_str(" facts={")?;
            for (index, (key, value)) in self.0.facts.iter().enumerate() {
                if index > 0 {
                    f.write_str(", ")?;
                }
                write!(f, "{key}={}", quote(value))?;
            }
            f.write_str("}")?;
        }

        if let Some(source) = &self.0.source {
            write!(f, " source={}", quote(&source.to_string()))?;
        }

        Ok(())
    }
}

impl fmt::Debug for QueryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QueryError")
            .field("query_id", &self.0.query_id)
            .field("trace_id", &self.0.trace_id)
            .field("code", &self.0.code)
            .field("kind", &self.kind())
            .field("status", &self.status())
            .field("operation", &self.0.operation)
            .field("message", &self.0.message)
            .field("facts", &self.0.facts)
            .field("cugraph_allocation_source", &self.0.cugraph_allocation_source)
            .field("graph_cache_scope_peak_bytes", &self.0.graph_cache_scope_peak_bytes)
            .field("source", &self.0.source.as_ref().map(|source| source.to_string()))
            .finish()
    }
}

impl StdError for QueryError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        self.source_error()
    }
}

fn quote(value: &str) -> String {
    if value.chars().all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.' | ':' | '/' | '(' | ')')) {
        value.to_string()
    } else {
        format!("{value:?}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redact_uri_tokens_replaces_uri_like_tokens_only() {
        assert_eq!(
            redact_uri_tokens("S3: must provide `aws_access_key` if AWS_ACCESS_KEY_ID isn't set."),
            "S3: must provide `aws_access_key` if AWS_ACCESS_KEY_ID isn't set."
        );
        assert_eq!(
            redact_uri_tokens("failed to open s3://bucket/secret/key.parquet for read"),
            "failed to open <redacted-uri> for read"
        );
        assert_eq!(redact_uri_tokens("  spaced   message  "), "spaced message");
    }

    #[cfg(feature = "cugraph")]
    #[test]
    fn graph_cache_scope_peak_is_a_typed_sideband_not_an_error_fact() {
        let error = QueryError::new(
            QueryErrorCode::CudfResourceExhausted,
            "query_engine.test.cache_scope",
            "cache scope failed",
        )
        .with_graph_cache_scope_peak_bytes(4_096);

        assert_eq!(error.graph_cache_scope_peak_bytes(), Some(4_096));
        assert_eq!(error.fact_value("graph_cache_scope_peak_bytes"), None);
        assert!(error.to_json()["facts"].get("graph_cache_scope_peak_bytes").is_none());
    }

    #[test]
    fn duplicate_fact_decoration_preserves_first_value_and_records_collision() {
        let error = QueryError::new(
            QueryErrorCode::CudfInvalidInput,
            "query_engine.test.fact_collision",
            "fact collision probe",
        )
        .with_fact("layer", "outer")
        .with_fact("layer", "inner")
        .with_fact("", "empty key is ignored");

        assert_eq!(error.fact_value("layer"), Some("outer"));
        assert_eq!(error.fact_value(FACT_COLLISION_KEYS), Some("<empty>,layer"));
        assert_eq!(error.code(), QueryErrorCode::CudfInvalidInput);
    }

    #[test]
    fn repeated_source_decoration_preserves_first_cause() {
        let first =
            QueryError::new(QueryErrorCode::CudfInvalidInput, "query_engine.test.first_cause", "the original cause");
        let expected_cause = first.to_string();
        let error = QueryError::new(
            QueryErrorCode::CudfInvalidInput,
            "query_engine.test.source_preservation",
            "source preservation probe",
        )
        .with_source(first)
        .with_source(QueryError::new(
            QueryErrorCode::DeviceFault,
            "query_engine.test.decoration",
            "decoration must not replace the cause",
        ));

        let cause = error.source_error().expect("first source must be retained").to_string();
        assert_eq!(cause, expected_cause, "{cause}");
        assert!(!cause.contains("decoration must not replace the cause"), "{cause}");
    }
}
