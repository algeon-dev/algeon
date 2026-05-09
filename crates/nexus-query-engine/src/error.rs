//! Error taxonomy for the DataFusion-free native query engine.
//!
//! Native errors stop at the `nexus-query-engine` API boundary as [`QueryError`].
//! The DataFusion adapter wraps them in its own `datafusion_nexus::Error` so it
//! can preserve native classification facts while exposing adapter-level
//! planning and execution categories to DataFusion.

use std::{collections::BTreeMap, error::Error as StdError, fmt};

use serde_json::{Value, json};

use crate::correlation::{QueryId, TraceId};

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
        Self::SourceAccessDenied,
        Self::SourceUnauthenticated,
        Self::SourceObjectNotFound,
        Self::SourceTemporaryFailure,
        Self::CudfInvariantViolation,
        Self::MemoryReservationFailed,
        Self::ServiceOverloaded,
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
            Self::SourceAccessDenied => "source_access_denied",
            Self::SourceUnauthenticated => "source_unauthenticated",
            Self::SourceObjectNotFound => "source_object_not_found",
            Self::SourceTemporaryFailure => "source_temporary_failure",
            Self::CudfInvariantViolation => "cudf_invariant_violation",
            Self::MemoryReservationFailed => "memory_reservation_failed",
            Self::ServiceOverloaded => "service_overloaded",
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
        | QueryErrorCode::ServiceOverloaded => QueryErrorKind::ResourceExhausted,
        QueryErrorCode::CudfTemporaryFailure
        | QueryErrorCode::CudfPersistentFailure
        | QueryErrorCode::CudfDependencyFailure
        | QueryErrorCode::SourceAccessDenied
        | QueryErrorCode::SourceUnauthenticated
        | QueryErrorCode::SourceObjectNotFound
        | QueryErrorCode::SourceTemporaryFailure => QueryErrorKind::DependencyFailure,
        QueryErrorCode::CudfInvariantViolation => QueryErrorKind::InvariantViolation,
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
        | QueryErrorCode::ServiceOverloaded => QueryErrorStatus::Temporary,
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
        | QueryErrorCode::SourceAccessDenied
        | QueryErrorCode::SourceUnauthenticated
        | QueryErrorCode::SourceObjectNotFound
        | QueryErrorCode::CudfInvariantViolation
        | QueryErrorCode::Internal => QueryErrorStatus::Permanent,
    }
}

/// Machine-readable facts attached to a [`QueryError`].
///
/// Fact keys are static strings so they can be treated as a documented contract
/// by tests, reports, and callers. Values are stringified with [`ToString`].
/// Iteration is sorted by key because facts are stored in a [`BTreeMap`].
///
/// Empty keys and duplicate inserts panic. The native engine reserves
/// `cudf_code`, `cudf_kind`, `cudf_status`, `cudf_operation`, and `cudf_message`
/// for wrapped `cudf::Error` facts.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Facts {
    values: BTreeMap<&'static str, String>,
}

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
        assert!(!key.is_empty(), "nexus-query-engine error fact key must not be empty");
        let previous = self.values.insert(key, value);
        assert!(previous.is_none(), "nexus-query-engine error fact key inserted twice: {key}");
    }

    fn set(&mut self, key: &'static str, value: String) {
        assert!(!key.is_empty(), "nexus-query-engine error fact key must not be empty");
        self.values.insert(key, value);
    }
}

/// Structured native-engine error with stable classification and facts.
pub struct QueryError {
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
    /// `nexus_query_engine.plan.execute`. `message` is human-facing diagnostic
    /// text. Add machine-readable context with [`Self::with_fact`].
    pub fn new(code: QueryErrorCode, operation: &'static str, message: impl Into<String>) -> Self {
        let message = message.into();
        assert!(!operation.is_empty(), "nexus-query-engine error operation must not be empty");
        assert!(!message.is_empty(), "nexus-query-engine error message must not be empty");
        Self {
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
        }
    }

    /// Marks an owner-reported cleanup wait as an elapsed deadline.
    ///
    /// This internal typed bit is deliberately separate from the public error
    /// code: `ServiceOverloaded` is also used by ordinary admission and cannot
    /// by itself authorize indefinite cleanup retries.
    #[must_use]
    pub(crate) fn with_cleanup_deadline_elapsed(mut self) -> Self {
        self.cleanup_disposition = Some(CleanupDisposition::DeadlineElapsed);
        self
    }

    pub(crate) fn is_cleanup_deadline_elapsed(&self) -> bool {
        self.cleanup_disposition == Some(CleanupDisposition::DeadlineElapsed)
    }

    /// Adds one machine-readable fact.
    ///
    /// `key` must be a non-empty static string and should be treated as part of
    /// the diagnostic contract once exposed. `value` is stringified with
    /// [`ToString`]. Duplicate keys panic.
    #[must_use]
    pub fn with_fact(mut self, key: &'static str, value: impl ToString) -> Self {
        self.facts.insert(key, value.to_string());
        self
    }

    #[must_use]
    pub(crate) fn with_fact_if_absent(mut self, key: &'static str, value: impl ToString) -> Self {
        if self.facts.get(key).is_none() {
            self.facts.insert(key, value.to_string());
        }
        self
    }

    /// Retains typed cuGraph allocation provenance and mirrors its stable label
    /// into ordinary error facts for existing diagnostic consumers.
    #[cfg(any(feature = "cugraph", test))]
    #[must_use]
    pub(crate) fn with_cugraph_allocation_source(mut self, source: CugraphAllocationSource) -> Self {
        self.facts.insert("cugraph_allocation_source", source.as_str().to_string());
        self.cugraph_allocation_source = Some(source);
        self
    }

    /// Returns the closed cuGraph allocation provenance, when the cuGraph
    /// boundary supplied one.
    pub const fn cugraph_allocation_source(&self) -> Option<CugraphAllocationSource> {
        self.cugraph_allocation_source
    }

    /// Retains an exact high-water mark from a completed resident-graph cache
    /// scope. This is an internal typed sideband, deliberately distinct from
    /// error facts so consumers cannot mistake diagnostic text for a peak.
    #[cfg(feature = "cugraph")]
    #[must_use]
    pub(crate) fn with_graph_cache_scope_peak_bytes(mut self, peak_bytes: u64) -> Self {
        self.graph_cache_scope_peak_bytes = Some(peak_bytes);
        self
    }

    /// Returns the exact resident-cache scope high-water carried from a graph
    /// execution error only after that scope completed its fence and health
    /// proof.
    #[cfg(feature = "cugraph")]
    pub(crate) const fn graph_cache_scope_peak_bytes(&self) -> Option<u64> {
        self.graph_cache_scope_peak_bytes
    }

    /// Assigns a query-wide diagnostic correlation id and mirrors it into facts.
    #[must_use]
    pub fn with_query_id(mut self, query_id: impl Into<QueryId>) -> Self {
        let query_id = query_id.into();
        self.facts.set("query_id", query_id.to_string());
        self.query_id = Some(query_id);
        self
    }

    /// Assigns an optional parent trace id and mirrors it into facts.
    #[must_use]
    pub fn with_trace_id(mut self, trace_id: impl Into<TraceId>) -> Self {
        let trace_id = trace_id.into();
        self.facts.set("trace_id", trace_id.to_string());
        self.trace_id = Some(trace_id);
        self
    }

    /// Attaches the source error, replacing any source already present.
    #[must_use]
    pub fn with_source<E>(mut self, source: E) -> Self
    where E: StdError + Send + Sync + 'static {
        self.source = Some(Box::new(source));
        self
    }

    /// Wraps a `cudf::Error` at the native-engine boundary.
    ///
    /// `operation` should be the stable dotted id for the native operation that
    /// called cuDF, and `message` should describe the failure for humans. cuDF
    /// classification and a URI-redacted message are preserved as `cudf_*`
    /// facts.
    pub fn from_cudf(operation: &'static str, message: impl Into<String>, source: cudf::Error) -> Self {
        let code = code_for_cudf_error(&source);
        let error = Self::new(code, operation, message)
            .with_fact("cudf_code", source.code())
            .with_fact("cudf_kind", source.kind())
            .with_fact("cudf_status", source.status())
            .with_fact("cudf_operation", source.operation());
        match cudf_message_fact(&source) {
            Some(cudf_message) => error.with_fact("cudf_message", cudf_message),
            None => error,
        }
        .with_source(source)
    }

    /// Wraps a `cuvs::Error` at the native vector-execution boundary.
    ///
    /// Classification is derived only from the typed cuVS kind and status;
    /// foreign message text remains diagnostic and never drives retry policy.
    #[cfg(feature = "cuvs")]
    pub fn from_cuvs(operation: &'static str, message: impl Into<String>, source: cuvs::Error) -> Self {
        let code = code_for_cuvs_error(&source);
        let error = Self::new(code, operation, message)
            .with_fact("cuvs_kind", source.kind())
            .with_fact("cuvs_status", source.status())
            .with_fact("cuvs_operation", source.operation());
        let cuvs_message = source.message().trim();
        let mut error = if cuvs_message.is_empty() {
            error
        } else {
            error.with_fact("cuvs_message", redact_uri_tokens(cuvs_message))
        };
        for (key, value) in source.context().filter(|(key, _)| is_safe_cuvs_context_fact(key)) {
            error = error.with_fact(key, value);
        }
        error.with_source(source)
    }

    /// Wraps a prepared cuDF interop failure at a native vector boundary.
    #[cfg(feature = "cuvs")]
    pub(crate) fn from_cudf_interop<E>(
        operation: &'static str,
        message: impl Into<String>,
        source_kind: impl Into<String>,
        source: E,
    ) -> Self
    where
        E: StdError + Send + Sync + 'static,
    {
        // The engine validates vector schemas and selects every requested
        // column before preparing this producer-rooted export. A preparation
        // failure at this point is therefore an execution-binding invariant,
        // not a second user-input validation surface.
        let source_kind: String = source_kind.into();
        Self::new(QueryErrorCode::CudfInvariantViolation, operation, message)
            .with_fact("cudf_interop_kind", source_kind)
            .with_source(source)
    }

    /// Wraps a `cudf::Error` without attaching its display text as a source.
    ///
    /// Use this when source text may contain sensitive reader locations. It
    /// keeps cuDF classification and redacted `cudf_message` facts, but omits
    /// the unsanitized source chain.
    pub(crate) fn from_cudf_without_source(
        operation: &'static str,
        message: impl Into<String>,
        source: &cudf::Error,
    ) -> Self {
        let code = code_for_cudf_error(source);
        let error = Self::new(code, operation, message)
            .with_fact("cudf_code", source.code())
            .with_fact("cudf_kind", source.kind())
            .with_fact("cudf_status", source.status())
            .with_fact("cudf_operation", source.operation());
        match cudf_message_fact(source) {
            Some(cudf_message) => error.with_fact("cudf_message", cudf_message),
            None => error,
        }
    }

    pub(crate) fn from_cudf_without_source_with_budget_details(
        operation: &'static str,
        message: impl Into<String>,
        source: &cudf::Error,
    ) -> Self {
        let budget_details = cudf_budget_details(source);
        let mut error = Self::from_cudf_without_source(operation, message, source);
        for (key, value) in budget_details {
            error = error.with_fact(key, value);
        }
        error
    }

    /// Creates a native-engine error from cuDF and preserves budget-token
    /// diagnostics when the lower layer attached them.
    pub fn from_cudf_with_budget_details(
        operation: &'static str,
        message: impl Into<String>,
        source: cudf::Error,
    ) -> Self {
        let budget_details = cudf_budget_details(&source);
        let mut error = Self::from_cudf(operation, message, source);
        for (key, value) in budget_details {
            error = error.with_fact(key, value);
        }
        error
    }

    /// Returns the stable machine-readable error code.
    pub fn code(&self) -> QueryErrorCode {
        self.code
    }

    /// Returns the coarse category derived from [`Self::code`].
    pub fn kind(&self) -> QueryErrorKind {
        self.code.kind()
    }

    /// Returns the retry/remediation status derived from [`Self::code`].
    pub fn status(&self) -> QueryErrorStatus {
        self.code.status()
    }

    /// Returns the query-wide diagnostic correlation id, when assigned.
    pub fn query_id(&self) -> Option<&QueryId> {
        self.query_id.as_ref()
    }

    /// Returns the optional parent trace id, when assigned.
    pub fn trace_id(&self) -> Option<&TraceId> {
        self.trace_id.as_ref()
    }

    /// Returns the stable dotted operation id that produced the error.
    pub fn operation(&self) -> &'static str {
        self.operation
    }

    /// Returns the human-facing diagnostic message.
    pub fn message(&self) -> &str {
        &self.message
    }

    /// Returns the sorted machine-readable fact set.
    pub fn facts(&self) -> &Facts {
        self.facts.as_ref()
    }

    pub fn fact_value(&self, key: &str) -> Option<&str> {
        self.facts.get(key)
    }

    /// Returns the underlying source error when one was attached.
    pub fn source_error(&self) -> Option<&(dyn StdError + 'static)> {
        self.source.as_deref().map(|source| source as &(dyn StdError + 'static))
    }

    /// Typed cuDF budget-exhaustion counters preserved from the source chain.
    pub fn cudf_budget_exhaustion(&self) -> Option<&cudf::BudgetExhaustionDetail> {
        let mut current = self.source_error();
        while let Some(error) = current {
            if let Some(cudf_error) = error.downcast_ref::<cudf::Error>()
                && let Some(detail) = cudf_error.budget_exhaustion()
            {
                return Some(detail);
            }
            current = error.source();
        }
        None
    }

    /// Serializes this error as a structured diagnostic JSON object.
    pub fn to_json(&self) -> Value {
        let query_id = self.query_id.as_ref().map(QueryId::as_str).or_else(|| self.fact_value("query_id"));
        let trace_id = self.trace_id.as_ref().map(TraceId::as_str).or_else(|| self.fact_value("trace_id"));
        json!({
            "schema": "nexus_query_engine.query_error.v1",
            "query_id": query_id,
            "trace_id": trace_id,
            "triage": self.triage_summary().to_json(),
            "code": self.code.as_str(),
            "kind": self.kind().as_str(),
            "status": self.status().as_str(),
            "operation": self.operation,
            "message": self.message,
            "facts": self.facts.iter().collect::<BTreeMap<_, _>>(),
            "source": self.source.as_ref().map(|source| source.to_string()),
        })
    }
}

/// Returns the trimmed foreign cuDF message, redacting URI-like tokens.
fn cudf_message_fact(source: &cudf::Error) -> Option<String> {
    let message = source.message().trim();
    if message.is_empty() {
        return None;
    }
    Some(redact_uri_tokens(message))
}

/// Redacts whitespace-delimited `scheme://...` tokens and normalizes spaces.
fn redact_uri_tokens(message: &str) -> String {
    message
        .split_whitespace()
        .map(|token| if token.contains("://") { "<redacted-uri>" } else { token })
        .collect::<Vec<_>>()
        .join(" ")
}

fn cudf_budget_details(source: &cudf::Error) -> Vec<(&'static str, String)> {
    source
        .details()
        .iter()
        .filter(|(key, _)| {
            matches!(
                *key,
                "budget_boundary"
                    | "budget_bytes"
                    | "budget_enforcement"
                    | "budget_status"
                    | "budget_peak_bytes"
                    | "budget_current_bytes"
                    | "budget_peak_aligned_bytes"
                    | "budget_current_aligned_bytes"
                    | "budget_failed_allocation_count"
                    | "budget_largest_failed_allocation_bytes"
            )
        })
        .map(|(key, value)| (*key, value.clone()))
        .collect()
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
            quote(&self.message),
        )?;

        if self.facts.values.is_empty() {
            f.write_str(" facts={}")?;
        } else {
            f.write_str(" facts={")?;
            for (index, (key, value)) in self.facts.iter().enumerate() {
                if index > 0 {
                    f.write_str(", ")?;
                }
                write!(f, "{key}={}", quote(value))?;
            }
            f.write_str("}")?;
        }

        if let Some(source) = &self.source {
            write!(f, " source={}", quote(&source.to_string()))?;
        }

        Ok(())
    }
}

impl fmt::Debug for QueryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QueryError")
            .field("query_id", &self.query_id)
            .field("trace_id", &self.trace_id)
            .field("code", &self.code)
            .field("kind", &self.kind())
            .field("status", &self.status())
            .field("operation", &self.operation)
            .field("message", &self.message)
            .field("facts", &self.facts)
            .field("cugraph_allocation_source", &self.cugraph_allocation_source)
            .field("graph_cache_scope_peak_bytes", &self.graph_cache_scope_peak_bytes)
            .field("source", &self.source.as_ref().map(|source| source.to_string()))
            .finish()
    }
}

impl StdError for QueryError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        self.source_error()
    }
}

// cuDF classification preserves permanent size/host-limit codes first, then
// generic resource exhaustion, retry status (`Temporary`/`Persistent`), and
// permanent failures by cuDF kind. This keeps
// cuDF's coarse status contract intact. Sealed attempt advice, not this
// status alone, governs retry; Persistent requires escalation or intervention.
// Notable permanent mappings:
// `NotFound` is treated as invalid input, `ExecutionContext` is treated as an
// invariant violation, and unknown kinds fall back to dependency failure.
fn code_for_cudf_error(error: &cudf::Error) -> QueryErrorCode {
    match error.code() {
        cudf::ErrorCode::SizeOverflow => return QueryErrorCode::SizeOverflow,
        cudf::ErrorCode::HostMaterializationLimitExceeded => {
            return QueryErrorCode::HostMaterializationLimitExceeded;
        },
        _ => {},
    }
    if error.kind() == cudf::ErrorKind::ResourceExhausted {
        return QueryErrorCode::CudfResourceExhausted;
    }

    match error.status() {
        cudf::ErrorStatus::Temporary => return QueryErrorCode::CudfTemporaryFailure,
        cudf::ErrorStatus::Persistent => return QueryErrorCode::CudfPersistentFailure,
        cudf::ErrorStatus::Permanent => {},
    }

    match error.kind() {
        cudf::ErrorKind::InvalidInput | cudf::ErrorKind::NotFound => QueryErrorCode::CudfInvalidInput,
        cudf::ErrorKind::Unsupported => QueryErrorCode::CudfUnsupported,
        cudf::ErrorKind::DependencyFailure => QueryErrorCode::CudfDependencyFailure,
        cudf::ErrorKind::InvariantViolation | cudf::ErrorKind::ExecutionContext => {
            QueryErrorCode::CudfInvariantViolation
        },
        _ => QueryErrorCode::CudfDependencyFailure,
    }
}

#[cfg(feature = "cuvs")]
fn code_for_cuvs_error(error: &cuvs::Error) -> QueryErrorCode {
    match error.kind() {
        cuvs::ErrorKind::InvalidInput => QueryErrorCode::CudfInvalidInput,
        cuvs::ErrorKind::Unsupported => QueryErrorCode::CudfUnsupported,
        cuvs::ErrorKind::OutOfMemory => QueryErrorCode::CudfResourceExhausted,
        cuvs::ErrorKind::InvalidState => QueryErrorCode::CudfInvariantViolation,
        cuvs::ErrorKind::Unexpected => match error.status() {
            cuvs::ErrorStatus::Temporary => QueryErrorCode::CudfTemporaryFailure,
            cuvs::ErrorStatus::Persistent => QueryErrorCode::CudfPersistentFailure,
            cuvs::ErrorStatus::Permanent => QueryErrorCode::CudfDependencyFailure,
        },
        _ => QueryErrorCode::CudfDependencyFailure,
    }
}

#[cfg(feature = "cuvs")]
fn is_safe_cuvs_context_fact(key: &str) -> bool {
    matches!(
        key,
        "cudf_code"
            | "cudf_kind"
            | "cudf_status"
            | "cudf_operation"
            | "interop_kind"
            | "native_status"
            | "buffer_binding"
            | "resource_binding"
            | "expected_device"
            | "actual_device"
    )
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

    fn sample_cudf_error() -> cudf::Error {
        cudf::io::parquet::RemoteParquetSource::s3("gs://bucket/key.parquet")
            .expect_err("non-s3 scheme must be rejected")
    }

    #[test]
    fn from_cudf_preserves_foreign_message_as_cudf_message_fact() {
        let source = sample_cudf_error();
        let expected_message = redact_uri_tokens(source.message().trim());

        let error = QueryError::from_cudf("nexus_query_engine.test.from_cudf", "wrapped cudf failure", source);

        assert_eq!(error.fact_value("cudf_message"), Some(expected_message.as_str()));
        assert!(error.source_error().is_some());
    }

    #[test]
    fn from_cudf_without_source_preserves_foreign_message_without_source_chain() {
        let source = sample_cudf_error();
        let expected_message = redact_uri_tokens(source.message().trim());

        let error = QueryError::from_cudf_without_source(
            "nexus_query_engine.test.from_cudf_without_source",
            "wrapped cudf failure",
            &source,
        );

        assert_eq!(error.fact_value("cudf_message"), Some(expected_message.as_str()));
        assert!(error.source_error().is_none());
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

    #[cfg(feature = "cugraph")]
    #[test]
    fn graph_cache_scope_peak_is_a_typed_sideband_not_an_error_fact() {
        let error = QueryError::new(
            QueryErrorCode::CudfResourceExhausted,
            "nexus_query_engine.test.cache_scope",
            "cache scope failed",
        )
        .with_graph_cache_scope_peak_bytes(4_096);

        assert_eq!(error.graph_cache_scope_peak_bytes(), Some(4_096));
        assert_eq!(error.fact_value("graph_cache_scope_peak_bytes"), None);
        assert!(error.to_json()["facts"].get("graph_cache_scope_peak_bytes").is_none());
    }

    #[cfg(feature = "cuvs")]
    #[test]
    fn from_cuvs_maps_only_typed_kind_and_status() {
        let cases = [
            (cuvs::Error::new(cuvs::ErrorKind::InvalidInput, "cuvs.invalid"), QueryErrorCode::CudfInvalidInput),
            (cuvs::Error::new(cuvs::ErrorKind::Unsupported, "cuvs.unsupported"), QueryErrorCode::CudfUnsupported),
            (cuvs::Error::new(cuvs::ErrorKind::OutOfMemory, "cuvs.oom"), QueryErrorCode::CudfResourceExhausted),
            (cuvs::Error::new(cuvs::ErrorKind::InvalidState, "cuvs.state"), QueryErrorCode::CudfInvariantViolation),
            (
                cuvs::Error::new(cuvs::ErrorKind::Unexpected, "cuvs.temporary")
                    .with_context("native_status", "CUVS_ERROR")
                    .set_temporary(),
                QueryErrorCode::CudfTemporaryFailure,
            ),
            (
                cuvs::Error::new(cuvs::ErrorKind::Unexpected, "cuvs.persistent").set_persistent(),
                QueryErrorCode::CudfPersistentFailure,
            ),
        ];
        for (source, expected) in cases {
            let error = QueryError::from_cuvs("nexus_query_engine.test.from_cuvs", "wrapped cuVS failure", source);
            assert_eq!(error.code(), expected);
            assert!(error.fact_value("cuvs_kind").is_some());
            assert!(error.fact_value("cuvs_status").is_some());
            assert!(error.fact_value("cuvs_operation").is_some());
            if error.fact_value("cuvs_operation") == Some("cuvs.temporary") {
                assert_eq!(error.fact_value("native_status"), Some("CUVS_ERROR"));
            }
        }
    }
}
