//! Error taxonomy for the DataFusion-free native query engine.
//!
//! Native errors stop at the `cudf-query-engine` API boundary as [`QueryError`].
//! The DataFusion adapter wraps them in its own `datafusion_cudf::Error` so it
//! can preserve native classification facts while exposing adapter-level
//! planning and execution categories to DataFusion.
#![deny(missing_docs)]

use std::{collections::BTreeMap, error::Error as StdError, fmt};

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
#[non_exhaustive]
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
    /// This is a temporary resource condition, separate from a cuDF allocator
    /// failure, and is intended to drive retry/backoff decisions.
    MemoryReservationFailed,
    /// An internal native-engine bug or unreachable condition.
    Internal,
}

impl QueryErrorCode {
    /// Stable snake-case code label used in reports and error text.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cancelled => "cancelled",
            Self::UnsupportedPlanShape => "unsupported_plan_shape",
            Self::UnsupportedOperator => "unsupported_operator",
            Self::UnsupportedExpression => "unsupported_expression",
            Self::UnsupportedType => "unsupported_type",
            Self::InvalidPlan => "invalid_plan",
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QueryErrorStatus {
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
        QueryErrorCode::CudfInvalidInput => QueryErrorKind::InvalidInput,
        QueryErrorCode::CudfUnsupported => QueryErrorKind::Unsupported,
        QueryErrorCode::CudfResourceExhausted | QueryErrorCode::MemoryReservationFailed => {
            QueryErrorKind::ResourceExhausted
        },
        QueryErrorCode::CudfTemporaryFailure
        | QueryErrorCode::CudfPersistentFailure
        | QueryErrorCode::CudfDependencyFailure => QueryErrorKind::DependencyFailure,
        QueryErrorCode::CudfInvariantViolation => QueryErrorKind::InvariantViolation,
        QueryErrorCode::Internal => QueryErrorKind::Internal,
    }
}

/// Maps each native error code to its stable retry/remediation status.
///
/// `status` answers what kind of remediation is plausible. `Permanent` means
/// retrying the same operation is not expected to help. `Persistent` preserves
/// cuDF's dependency/runtime status and is deliberately not a synonym for
/// `Permanent`.
pub const fn status_for_code(code: QueryErrorCode) -> QueryErrorStatus {
    match code {
        QueryErrorCode::Cancelled => QueryErrorStatus::Permanent,
        QueryErrorCode::CudfResourceExhausted
        | QueryErrorCode::CudfTemporaryFailure
        | QueryErrorCode::MemoryReservationFailed => QueryErrorStatus::Temporary,
        QueryErrorCode::CudfPersistentFailure => QueryErrorStatus::Persistent,
        QueryErrorCode::UnsupportedPlanShape
        | QueryErrorCode::UnsupportedOperator
        | QueryErrorCode::UnsupportedExpression
        | QueryErrorCode::UnsupportedType
        | QueryErrorCode::InvalidPlan
        | QueryErrorCode::CudfInvalidInput
        | QueryErrorCode::CudfUnsupported
        | QueryErrorCode::CudfDependencyFailure
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
/// Empty keys panic, and inserting the same key twice panics. The native engine
/// reserves `cudf_code`, `cudf_kind`, `cudf_status`, and `cudf_operation` for
/// facts copied from an underlying `cudf::Error`.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Facts {
    values: BTreeMap<&'static str, String>,
}

impl Facts {
    /// Creates an empty fact set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the string value for `key` when present.
    pub fn get(&self, key: &str) -> Option<&str> {
        self.values.get(key).map(String::as_str)
    }

    /// Iterates facts in sorted key order.
    pub fn iter(&self) -> impl Iterator<Item = (&'static str, &str)> {
        self.values.iter().map(|(key, value)| (*key, value.as_str()))
    }

    fn insert(&mut self, key: &'static str, value: String) {
        assert!(!key.is_empty(), "cudf-query-engine error fact key must not be empty");
        let previous = self.values.insert(key, value);
        assert!(previous.is_none(), "cudf-query-engine error fact key inserted twice: {key}");
    }
}

/// Structured native-engine error with stable classification and facts.
pub struct QueryError {
    code: QueryErrorCode,
    operation: &'static str,
    message: String,
    facts: Facts,
    source: Option<BoxedSourceError>,
}

impl QueryError {
    /// Creates a native-engine error with a stable operation id and message.
    ///
    /// `operation` should be a stable dotted identifier such as
    /// `cudf_query_engine.plan.execute`. `message` is human-facing diagnostic
    /// text. Add machine-readable context with [`Self::with_fact`].
    pub fn new(code: QueryErrorCode, operation: &'static str, message: impl Into<String>) -> Self {
        let message = message.into();
        assert!(!operation.is_empty(), "cudf-query-engine error operation must not be empty");
        assert!(!message.is_empty(), "cudf-query-engine error message must not be empty");
        Self { code, operation, message, facts: Facts::new(), source: None }
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
    /// called cuDF, and `message` should describe the failure for humans. The
    /// cuDF code, kind, status, and operation are preserved as reserved facts:
    /// `cudf_code`, `cudf_kind`, `cudf_status`, and `cudf_operation`.
    pub fn from_cudf(operation: &'static str, message: impl Into<String>, source: cudf::Error) -> Self {
        let code = code_for_cudf_error(&source);
        Self::new(code, operation, message)
            .with_fact("cudf_code", source.code())
            .with_fact("cudf_kind", source.kind())
            .with_fact("cudf_status", source.status())
            .with_fact("cudf_operation", source.operation())
            .with_source(source)
    }

    /// Creates a native-engine error from cuDF and preserves budget-token
    /// diagnostics when the lower layer attached them.
    pub fn from_cudf_with_budget_details(
        operation: &'static str,
        message: impl Into<String>,
        source: cudf::Error,
    ) -> Self {
        let budget_details = source
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
            .collect::<Vec<_>>();
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
        &self.facts
    }

    /// Returns one fact value by key.
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
            .field("code", &self.code)
            .field("kind", &self.kind())
            .field("status", &self.status())
            .field("operation", &self.operation)
            .field("message", &self.message)
            .field("facts", &self.facts)
            .field("source", &self.source.as_ref().map(|source| source.to_string()))
            .finish()
    }
}

impl StdError for QueryError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        self.source_error()
    }
}

// cuDF classification priority is resource exhaustion first, then retry status
// (`Temporary`/`Persistent`), then permanent failures by cuDF kind. Notable
// permanent mappings: `NotFound` is treated as invalid input,
// `ExecutionContext` is treated as an invariant violation, and unknown kinds
// fall back to dependency failure.
fn code_for_cudf_error(error: &cudf::Error) -> QueryErrorCode {
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

fn quote(value: &str) -> String {
    if value.chars().all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.' | ':' | '/' | '(' | ')')) {
        value.to_string()
    } else {
        format!("{value:?}")
    }
}
