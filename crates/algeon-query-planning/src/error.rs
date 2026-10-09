//! Error taxonomy for the native query engine.
//!
//! Native errors stop at the `query-engine` API boundary as [`QueryError`].
//! The DataFusion adapter wraps them in its own `algeon_datafusion::Error` so it
//! can preserve native classification facts while exposing adapter-level
//! planning and execution categories to DataFusion.

use std::{
    borrow::Cow,
    collections::{BTreeMap, BTreeSet},
    error::Error as StdError,
    fmt,
    sync::Arc,
};

use serde::{Deserialize, Deserializer, Serialize, Serializer, de::Error as _};
use serde_json::{Value, json};

pub use self::context::{AllocationSource, FailureContext, MemoryFailure, MemoryUsage, ResourceBoundary};
use crate::{
    correlation::{QueryId, TraceId},
    triage::TriageCode,
};

mod context;

// Shared so an error can be cloned: one failure may be reported to every
// party waiting on the operation that failed.
type SharedSourceError = Arc<dyn StdError + Send + Sync + 'static>;

/// Native query-engine result type.
pub type Result<T> = std::result::Result<T, QueryError>;

/// Stable, machine-readable native-engine error codes.
///
/// Codes describe the closest actionable cause. [`QueryErrorKind`] and
/// [`QueryErrorStatus`] are derived from the code; consumers should not infer
/// those policies from variant names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum QueryErrorCode {
    /// Query execution was cancelled by the caller.
    Cancelled,
    /// A compiled pipeline contract refuses a plan that must execute natively.
    ///
    /// This is a capability rejection. A malformed plan is reported as `InvalidPlan`.
    UnsupportedPipelineContract,
    /// A specific operator within an otherwise recognizable plan is unsupported.
    UnsupportedOperator,
    /// An expression or expression form is unsupported.
    UnsupportedExpression,
    /// A data type, physical representation, or type combination is unsupported.
    UnsupportedType,
    /// The native IR violates executor invariants or cannot be validated.
    ///
    /// Use this when the plan is malformed for the native engine. A well-formed
    /// plan outside the supported catalogue is reported with an `Unsupported*` code.
    InvalidPlan,
    /// Caller-supplied execution-attempt correlation is invalid or oversized.
    InvalidAttemptCorrelation,
    /// A service, backend, or device configuration value is invalid or
    /// inconsistent with the discovered hardware.
    ///
    /// Unlike [`Self::InvalidPlan`], the input is deployment configuration, not
    /// a query plan. The same configuration fails again until it changes.
    InvalidConfiguration,
    /// Runtime graph inputs violate an engine-owned graph-operation contract.
    ///
    /// This code describes materialized edge or side-input data. `InvalidPlan`
    /// describes malformed native IR.
    GraphInputInvalid,
    /// cuDF rejected an input, including permanent `NotFound` conditions.
    CudfInvalidInput,
    /// cuDF reported a permanent unsupported capability.
    CudfUnsupported,
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
        Self::UnsupportedPipelineContract,
        Self::UnsupportedOperator,
        Self::UnsupportedExpression,
        Self::UnsupportedType,
        Self::InvalidPlan,
        Self::InvalidAttemptCorrelation,
        Self::InvalidConfiguration,
        Self::GraphInputInvalid,
        Self::CudfInvalidInput,
        Self::CudfUnsupported,
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
            Self::UnsupportedPipelineContract => "unsupported_pipeline_contract",
            Self::UnsupportedOperator => "unsupported_operator",
            Self::UnsupportedExpression => "unsupported_expression",
            Self::UnsupportedType => "unsupported_type",
            Self::InvalidPlan => "invalid_plan",
            Self::InvalidAttemptCorrelation => "invalid_attempt_correlation",
            Self::InvalidConfiguration => "invalid_configuration",
            Self::GraphInputInvalid => "graph_input_invalid",
            Self::CudfInvalidInput => "cudf_invalid_input",
            Self::CudfUnsupported => "cudf_unsupported",
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
    pub(crate) const fn as_str(self) -> &'static str {
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
/// This is a coarse source classification. Attempt retry advice is derived
/// only after terminal evidence is sealed.
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
    /// This remains distinct from [`Self::Permanent`]. It records a continuing
    /// dependency failure, while `Permanent` records a request that must change.
    /// Neither status alone is retry advice.
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
pub(crate) const fn kind_for_code(code: QueryErrorCode) -> QueryErrorKind {
    match code {
        QueryErrorCode::Cancelled => QueryErrorKind::Cancelled,
        QueryErrorCode::UnsupportedPipelineContract
        | QueryErrorCode::UnsupportedOperator
        | QueryErrorCode::UnsupportedExpression
        | QueryErrorCode::UnsupportedType => QueryErrorKind::Unsupported,
        QueryErrorCode::InvalidPlan => QueryErrorKind::Planning,
        QueryErrorCode::InvalidAttemptCorrelation => QueryErrorKind::InvalidInput,
        QueryErrorCode::InvalidConfiguration => QueryErrorKind::InvalidInput,
        QueryErrorCode::GraphInputInvalid => QueryErrorKind::InvalidInput,
        QueryErrorCode::CudfInvalidInput => QueryErrorKind::InvalidInput,
        QueryErrorCode::CudfUnsupported => QueryErrorKind::Unsupported,
        QueryErrorCode::SizeOverflow
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
pub(crate) const fn status_for_code(code: QueryErrorCode) -> QueryErrorStatus {
    match code {
        QueryErrorCode::Cancelled => QueryErrorStatus::Permanent,
        QueryErrorCode::CudfTemporaryFailure
        | QueryErrorCode::SourceTemporaryFailure
        | QueryErrorCode::MemoryReservationFailed
        | QueryErrorCode::ServiceOverloaded
        | QueryErrorCode::QueryBudgetUnavailable
        | QueryErrorCode::DeviceOutOfMemory
        | QueryErrorCode::DeviceAllocationFailed => QueryErrorStatus::Temporary,
        QueryErrorCode::CudfPersistentFailure => QueryErrorStatus::Persistent,
        QueryErrorCode::UnsupportedPipelineContract
        | QueryErrorCode::UnsupportedOperator
        | QueryErrorCode::UnsupportedExpression
        | QueryErrorCode::UnsupportedType
        | QueryErrorCode::InvalidPlan
        | QueryErrorCode::InvalidAttemptCorrelation
        | QueryErrorCode::InvalidConfiguration
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

pub(crate) const fn triage_code_for_query_error(code: QueryErrorCode) -> TriageCode {
    match code {
        QueryErrorCode::Cancelled => TriageCode::Cancelled,
        QueryErrorCode::UnsupportedPipelineContract => TriageCode::UnsupportedPlanShape,
        QueryErrorCode::UnsupportedOperator => TriageCode::UnsupportedOperator,
        QueryErrorCode::UnsupportedExpression => TriageCode::UnsupportedExpression,
        QueryErrorCode::UnsupportedType => TriageCode::UnsupportedType,
        QueryErrorCode::InvalidPlan => TriageCode::InvalidPlan,
        QueryErrorCode::InvalidAttemptCorrelation
        | QueryErrorCode::InvalidConfiguration
        | QueryErrorCode::GraphInputInvalid
        | QueryErrorCode::CudfInvalidInput => TriageCode::InvalidInput,
        QueryErrorCode::CudfUnsupported => TriageCode::UnsupportedOperator,
        QueryErrorCode::SizeOverflow
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

/// Failure code of a sealed attempt: a native error's own code, or one of the
/// attempt lifecycle outcomes no `QueryError` carries.
///
/// The wire label is the native code's label and `Debug` prints the native
/// code's name, so wrapping changes no report value.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AttemptFailureCode {
    /// Failure classified by the canonical native query error taxonomy.
    Query(QueryErrorCode),
    /// The backend was closing when the attempt needed service.
    BackendClosing,
    /// A DataFusion-owned failure terminated the attempt.
    DatafusionError,
    /// The result consumer dropped the attempt before normal completion.
    Dropped,
    /// Attempt execution unwound after a panic.
    Panicked,
    /// Stream completion could not establish safe resource release.
    StreamSynchronizationFailed,
    /// A cleanup callback was abandoned before it could report completion.
    CleanupCallbackAbandoned,
}

impl AttemptFailureCode {
    const LIFECYCLE: [Self; 6] = [
        Self::BackendClosing,
        Self::DatafusionError,
        Self::Dropped,
        Self::Panicked,
        Self::StreamSynchronizationFailed,
        Self::CleanupCallbackAbandoned,
    ];

    /// Wraps a native error code as an attempt failure without changing its classification.
    pub const fn from_query_error_code(code: QueryErrorCode) -> Self {
        Self::Query(code)
    }

    /// Stable label used by attempt reports for this failure code.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Query(code) => code.as_str(),
            Self::BackendClosing => "backend_closing",
            Self::DatafusionError => "datafusion_error",
            Self::Dropped => "dropped",
            Self::Panicked => "panicked",
            Self::StreamSynchronizationFailed => "stream_synchronization_failed",
            Self::CleanupCallbackAbandoned => "cleanup_callback_abandoned",
        }
    }

    fn from_label(label: &str) -> Option<Self> {
        QueryErrorCode::ALL
            .iter()
            .map(|code| Self::Query(*code))
            .chain(Self::LIFECYCLE)
            .find(|code| code.as_str() == label)
    }
}

impl fmt::Debug for AttemptFailureCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Query(code) => fmt::Debug::fmt(code, f),
            Self::BackendClosing => f.write_str("BackendClosing"),
            Self::DatafusionError => f.write_str("DatafusionError"),
            Self::Dropped => f.write_str("Dropped"),
            Self::Panicked => f.write_str("Panicked"),
            Self::StreamSynchronizationFailed => f.write_str("StreamSynchronizationFailed"),
            Self::CleanupCallbackAbandoned => f.write_str("CleanupCallbackAbandoned"),
        }
    }
}

impl Serialize for AttemptFailureCode {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for AttemptFailureCode {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        let label = <Cow<'de, str>>::deserialize(deserializer)?;
        Self::from_label(&label).ok_or_else(|| D::Error::custom(format!("unknown attempt failure code `{label}`")))
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
/// `fact_collision_keys` so a colliding decoration stays diagnosable without
/// replacing the primary error's facts. Callers that own a key and must
/// overwrite it use [`QueryError::with_query_id`], [`QueryError::with_trace_id`],
/// or another explicit overwrite path, never a second [`QueryError::with_fact`].
/// The native engine reserves `cudf_code`, `cudf_kind`, `cudf_status`,
/// `cudf_operation`, and `cudf_message` for wrapped `cudf::Error` facts.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Facts {
    values: BTreeMap<&'static str, String>,
    collision_keys: BTreeSet<&'static str>,
}

/// Fact key recording rejected duplicate or empty decoration keys.
pub(crate) const FACT_COLLISION_KEYS: &str = "fact_collision_keys";

/// Placeholder recorded when a decoration key is empty.
pub(crate) const EMPTY_FACT_KEY: &str = "<empty>";

impl Facts {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Looks up a stored fact by name, returning `None` when absent.
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
#[derive(Clone)]
pub struct QueryError(Box<QueryErrorRepr>);

#[derive(Clone)]
struct QueryErrorRepr {
    query_id: Option<QueryId>,
    trace_id: Option<TraceId>,
    code: QueryErrorCode,
    operation: &'static str,
    message: String,
    facts: Box<Facts>,
    context: FailureContext,
    source: Option<SharedSourceError>,
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
    /// [`Self::with_source`]) never panics and keeps the first value.
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
            context: FailureContext::default(),
            source: None,
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
        self.0.context.cleanup_deadline_elapsed = true;
        self
    }

    #[doc(hidden)]
    pub fn is_cleanup_deadline_elapsed(&self) -> bool {
        self.0.context.cleanup_deadline_elapsed
    }

    /// Adds one machine-readable fact.
    ///
    /// `key` must be a non-empty static string and should be treated as part of
    /// the diagnostic contract once exposed. `value` is stringified with
    /// [`ToString`]. A duplicate or empty key never panics and never replaces
    /// the primary error's facts. The first value wins, and the rejected key is
    /// recorded under the `fact_collision_keys` fact. Layered `with_fact` calls must
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
        if self.0.facts.get(key).is_none() && !self.0.context.projects(key) {
            self.0.facts.insert(key, value.to_string());
        }
        self
    }

    /// Returns the typed failure evidence that decisions read.
    pub fn context(&self) -> &FailureContext {
        &self.0.context
    }

    /// Records the budget scope the failure passed through. The innermost
    /// scope wins; a different later scope is reported as a fact collision.
    #[must_use]
    pub fn with_memory_failure(mut self, memory: MemoryFailure) -> Self {
        let candidate = FailureContext { memory: Some(memory), ..FailureContext::default() };
        match &self.0.context.memory {
            None => self.0.context.memory = candidate.memory,
            Some(_) => self.record_context_collisions(&candidate),
        }
        self
    }

    /// Records the execution boundary that owned the failed work. The first
    /// complete boundary wins as a unit, so an outer scope never pairs its
    /// node or side with an inner failure location.
    #[must_use]
    pub fn with_resource_boundary(mut self, boundary: ResourceBoundary) -> Self {
        let candidate = FailureContext { boundary: Some(boundary), ..FailureContext::default() };
        match &self.0.context.boundary {
            None => self.0.context.boundary = candidate.boundary,
            Some(_) => self.record_context_collisions(&candidate),
        }
        self
    }

    /// Records the native allocator exception family of an allocation failure.
    #[must_use]
    #[doc(hidden)]
    pub fn with_allocation_source(mut self, source: AllocationSource) -> Self {
        let candidate = FailureContext { allocation_source: Some(source), ..FailureContext::default() };
        match self.0.context.allocation_source {
            None => self.0.context.allocation_source = candidate.allocation_source,
            Some(_) => self.record_context_collisions(&candidate),
        }
        self
    }

    /// Retains an exact high-water mark from a completed resident-graph cache
    /// scope. It is typed evidence, never a wire fact, so consumers cannot
    /// mistake diagnostic text for a peak.
    #[must_use]
    #[doc(hidden)]
    pub fn with_graph_cache_scope_peak_bytes(mut self, peak_bytes: u64) -> Self {
        self.0.context.graph_cache_scope_peak_bytes.get_or_insert(peak_bytes);
        self
    }

    fn record_context_collisions(&mut self, candidate: &FailureContext) {
        let current = self.0.context.projection();
        for (key, value) in candidate.projection() {
            if current.iter().any(|(existing, current)| *existing == key && *current != value) {
                self.0.facts.record_collision(key);
            }
        }
    }

    /// Stored diagnostic facts plus the typed context rendered under its
    /// wire keys, sorted by key.
    ///
    /// The typed value wins over a stored fact with the same key; a stored
    /// fact that disagrees is reported under `fact_collision_keys`.
    pub fn wire_facts(&self) -> impl Iterator<Item = (&'static str, Cow<'_, str>)> {
        let mut facts = self
            .0
            .facts
            .iter()
            .filter(|(key, _)| *key != FACT_COLLISION_KEYS)
            .map(|(key, value)| (key, Cow::Borrowed(value)))
            .collect::<BTreeMap<_, _>>();
        let mut collisions = self.0.facts.collision_keys.clone();
        for (key, value) in self.0.context.projection() {
            if facts.insert(key, value.clone()).is_some_and(|stored| stored != value) {
                collisions.insert(key);
            }
        }
        if !collisions.is_empty() {
            facts.insert(FACT_COLLISION_KEYS, Cow::Owned(collisions.into_iter().collect::<Vec<_>>().join(",")));
        }
        facts.into_iter()
    }

    /// One value of [`Self::wire_facts`].
    pub fn wire_fact(&self, key: &str) -> Option<Cow<'_, str>> {
        self.wire_facts().find_map(|(candidate, value)| (candidate == key).then_some(value))
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
    /// in a new [`QueryError`] and do not re-attach a source.
    #[must_use]
    pub fn with_source<E>(mut self, source: E) -> Self
    where E: StdError + Send + Sync + 'static {
        if self.0.source.is_none() {
            self.0.source = Some(Arc::new(source));
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

    /// Returns a stored fact by name. Typed failure-context projections are
    /// available through [`Self::wire_facts`].
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
            "schema": "query_engine.query_error",
            "query_id": query_id,
            "trace_id": trace_id,
            "triage": self.triage_summary().to_json(),
            "code": self.0.code.as_str(),
            "kind": self.kind().as_str(),
            "status": self.status().as_str(),
            "operation": self.0.operation,
            "message": self.0.message,
            "facts": self.wire_facts().collect::<BTreeMap<_, _>>(),
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

/// Renders a byte count as exact bytes followed by GiB truncated to two
/// decimals, such as `101973819392 bytes (94.97 GiB)`.
///
/// Capacity and cap messages share this form so the exact value that must be
/// compared with a limit and the figure an operator recognizes appear
/// together. Truncation keeps a limit from reading larger than it is.
pub fn bytes_with_gib(bytes: u64) -> String {
    const GIB: u128 = 1 << 30;
    let hundredths = u128::from(bytes) * 100 / GIB;
    format!("{bytes} bytes ({}.{:02} GiB)", hundredths / 100, hundredths % 100)
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

        f.write_str(" facts={")?;
        for (index, (key, value)) in self.wire_facts().enumerate() {
            if index > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{key}={}", quote(&value))?;
        }
        f.write_str("}")?;

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
            .field("context", &self.0.context)
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
    use crate::plan::NodeId;

    #[test]
    fn bytes_with_gib_prints_exact_bytes_and_truncated_gib() {
        assert_eq!(bytes_with_gib(0), "0 bytes (0.00 GiB)");
        assert_eq!(bytes_with_gib(700), "700 bytes (0.00 GiB)");
        assert_eq!(bytes_with_gib(95 * (1 << 30)), "102005473280 bytes (95.00 GiB)");
        assert_eq!(bytes_with_gib(101_973_819_392), "101973819392 bytes (94.97 GiB)");
        assert_eq!(bytes_with_gib(95 * (1 << 30) - 1), "102005473279 bytes (94.99 GiB)");
        assert_eq!(bytes_with_gib(u64::MAX), "18446744073709551615 bytes (17179869183.99 GiB)");
    }

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

    #[test]
    fn graph_cache_scope_peak_is_a_typed_sideband_not_an_error_fact() {
        let error = QueryError::new(
            QueryErrorCode::DeviceAllocationFailed,
            "query_engine.test.cache_scope",
            "cache scope failed",
        )
        .with_graph_cache_scope_peak_bytes(4_096);

        assert_eq!(error.context().graph_cache_scope_peak_bytes, Some(4_096));
        assert_eq!(error.fact_value("graph_cache_scope_peak_bytes"), None);
        assert!(error.to_json()["facts"].get("graph_cache_scope_peak_bytes").is_none());
    }

    #[test]
    fn wire_facts_render_typed_context_over_disagreeing_stored_facts() {
        let node = NodeId::from_index_for_test(4);
        let error = QueryError::new(
            QueryErrorCode::QueryMemoryLimitExceeded,
            "query_engine.test.wire_facts",
            "typed context projection probe",
        )
        .with_fact("budget_bytes", 1)
        .with_fact("node_id", node)
        .with_memory_failure(MemoryFailure {
            boundary: "inner".into(),
            operation: "cudf.test.op".into(),
            usage: Some(MemoryUsage {
                limit_bytes: 4096,
                peak_bytes: 3,
                peak_aligned_bytes: 256,
                current_bytes: 2,
                current_aligned_bytes: 256,
                failed_allocation_count: 1,
                largest_failed_allocation_bytes: 8192,
            }),
        })
        .with_memory_failure(MemoryFailure { boundary: "outer".into(), operation: "cudf.test.op".into(), usage: None })
        .with_resource_boundary(ResourceBoundary::new("source_chunk_read", node).with_resume("source_chunk"))
        .with_resource_boundary(ResourceBoundary::new("outer_stage", node))
        .with_fact_if_absent("resume_boundary", "outer_resume");

        let wire = error.wire_facts().map(|(key, value)| (key, value.into_owned())).collect::<BTreeMap<_, _>>();
        assert_eq!(wire["budget_bytes"], "4096", "the allocator limit wins over a stale stored fact");
        assert_eq!(wire["budget_boundary"], "inner", "the innermost scope wins");
        assert_eq!(wire["memory_boundary"], "source_chunk_read", "the first boundary wins as a unit");
        assert_eq!(wire["resume_boundary"], "source_chunk", "a typed key counts as present");
        assert_eq!(wire["node_id"], "n4");
        assert_eq!(wire[FACT_COLLISION_KEYS], "budget_boundary,budget_bytes,memory_boundary,stage");
        assert_eq!(error.wire_facts().collect::<Vec<_>>(), error.wire_facts().collect::<Vec<_>>());
        assert!(error.to_string().contains("budget_bytes=4096"), "{error}");
    }

    #[test]
    fn attempt_failure_codes_round_trip_through_their_labels() {
        let codes = QueryErrorCode::ALL
            .iter()
            .map(|code| AttemptFailureCode::Query(*code))
            .chain(AttemptFailureCode::LIFECYCLE)
            .collect::<Vec<_>>();
        for code in codes {
            let json = serde_json::to_string(&code).unwrap();
            assert_eq!(json, format!("\"{}\"", code.as_str()));
            assert_eq!(serde_json::from_str::<AttemptFailureCode>(&json).unwrap(), code);
        }
        assert_eq!(AttemptFailureCode::Query(QueryErrorCode::Internal).as_str(), "internal");
        assert!(serde_json::from_str::<AttemptFailureCode>("\"unsupported_plan_shape\"").is_err());
    }

    #[test]
    fn invalid_configuration_is_a_permanent_invalid_input_code_distinct_from_invalid_plan() {
        let code = QueryErrorCode::InvalidConfiguration;
        assert_eq!(code.as_str(), "invalid_configuration");
        assert_eq!(code.kind(), QueryErrorKind::InvalidInput);
        assert_eq!(code.status(), QueryErrorStatus::Permanent);
        assert_eq!(TriageCode::from(code), TriageCode::InvalidInput);
        assert_eq!(QueryErrorCode::InvalidPlan.kind(), QueryErrorKind::Planning);
        assert!(QueryErrorCode::ALL.contains(&code));
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
