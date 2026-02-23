//! Error taxonomy for the DataFusion adapter boundary.
//!
//! Adapter errors classify failures from DataFusion planning/execution,
//! native execution, cuDF calls made directly by the adapter, and native
//! `cudf-query-engine` errors wrapped by [`Error::from_native`]. Native errors
//! keep their original classification in `native_*` facts while this layer maps
//! them into adapter-level [`ErrorCode`] values suitable for DataFusion.

use std::{collections::BTreeMap, error::Error as StdError, fmt};

use datafusion::common::DataFusionError;

mod code;

pub use code::{ErrorCode, ErrorKind, ErrorStatus, kind_for_code, status_for_code};

type BoxedSourceError = Box<dyn StdError + Send + Sync + 'static>;

/// Adapter result type used before conversion to `DataFusionError`.
pub type Result<T> = std::result::Result<T, Error>;

/// Machine-readable facts attached to an [`Error`].
///
/// Fact keys are static strings so they can be treated as a documented contract
/// by tests, reports, and callers. Values are stringified with [`ToString`].
/// Iteration is sorted by key because facts are stored in a [`BTreeMap`].
///
/// Empty keys panic, and inserting the same key twice panics. The adapter
/// uses `cudf_code`, `cudf_kind`, `cudf_status`, and `cudf_operation` for cuDF
/// source classification. Those keys may come from direct adapter cuDF calls or
/// from native facts copied unchanged by [`Error::from_native`]. The
/// `native_code`, `native_kind`, `native_status`, and `native_operation` keys
/// describe errors wrapped from `cudf-query-engine`.
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
        assert!(!key.is_empty(), "datafusion-cudf error fact key must not be empty");
        let previous = self.values.insert(key, value);
        assert!(previous.is_none(), "datafusion-cudf error fact key inserted twice: {key}");
    }
}

/// Structured adapter error with stable classification and facts.
pub struct Error {
    code: ErrorCode,
    operation: &'static str,
    message: String,
    facts: Facts,
    source: Option<BoxedSourceError>,
}

impl Error {
    /// Creates an adapter error with a stable operation id and message.
    ///
    /// `operation` should be a stable dotted identifier such as
    /// `datafusion_cudf.native_exec.execute`. `message` is human-facing
    /// diagnostic text. Add machine-readable context with [`Self::with_fact`].
    pub fn new(code: ErrorCode, operation: &'static str, message: impl Into<String>) -> Self {
        let message = message.into();
        assert!(!operation.is_empty(), "datafusion-cudf error operation must not be empty");
        assert!(!message.is_empty(), "datafusion-cudf error message must not be empty");
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

    /// Attaches the source error, replacing any source already present.
    #[must_use]
    pub fn with_source<E>(mut self, source: E) -> Self
    where E: StdError + Send + Sync + 'static {
        self.source = Some(Box::new(source));
        self
    }

    /// Wraps a `cudf::Error` at the adapter boundary.
    ///
    /// `operation` should be the stable dotted id for the adapter operation that
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

    /// Wraps a native-engine error at the DataFusion adapter boundary.
    ///
    /// The adapter maps the native code into the closest [`ErrorCode`] and
    /// preserves the native classification under reserved facts: `native_code`,
    /// `native_kind`, `native_status`, and `native_operation`. Native facts are
    /// copied unprefixed so existing machine-readable context is not hidden.
    /// Collisions with already-inserted adapter/native classification keys,
    /// especially the `native_*` keys, or duplicate native fact keys currently
    /// panic through [`Facts`] insertion.
    ///
    /// `operation` should be the stable dotted id for the adapter operation, and
    /// `message` should be human-facing adapter context.
    pub fn from_native(
        operation: &'static str,
        message: impl Into<String>,
        source: cudf_query_engine::QueryError,
    ) -> Self {
        let code = code_for_native_error(&source);
        let mut error = Self::new(code, operation, message)
            .with_fact("native_code", source.code())
            .with_fact("native_kind", source.kind())
            .with_fact("native_status", source.status())
            .with_fact("native_operation", source.operation());
        for (key, value) in source.facts().iter() {
            error.facts.insert(key, value.to_string());
        }
        error.with_source(source)
    }

    /// Returns the stable machine-readable adapter error code.
    pub fn code(&self) -> ErrorCode {
        self.code
    }

    /// Returns the coarse category derived from [`Self::code`].
    pub fn kind(&self) -> ErrorKind {
        self.code.kind()
    }

    /// Returns the retry/remediation status derived from [`Self::code`].
    pub fn status(&self) -> ErrorStatus {
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
}

impl fmt::Display for Error {
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

impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Error")
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

impl StdError for Error {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        self.source_error()
    }
}

// Adapter errors cross the DataFusion boundary as `External` so callers can
// downcast the boxed source back to `datafusion_cudf::Error` and inspect code,
// kind, status, operation, facts, and source.
impl From<Error> for DataFusionError {
    fn from(error: Error) -> Self {
        Self::External(Box::new(error))
    }
}

// cuDF classification priority is resource exhaustion first, then retry status
// (`Temporary`/`Persistent`), then permanent failures by cuDF kind. Notable
// permanent mappings: `NotFound` is treated as invalid input,
// `ExecutionContext` is treated as an invariant violation, and unknown kinds
// fall back to dependency failure.
fn code_for_cudf_error(error: &cudf::Error) -> ErrorCode {
    if error.kind() == cudf::ErrorKind::ResourceExhausted {
        return ErrorCode::CudfResourceExhausted;
    }

    match error.status() {
        cudf::ErrorStatus::Temporary => return ErrorCode::CudfTemporaryFailure,
        cudf::ErrorStatus::Persistent => return ErrorCode::CudfPersistentFailure,
        cudf::ErrorStatus::Permanent => {},
    }

    match error.kind() {
        cudf::ErrorKind::InvalidInput | cudf::ErrorKind::NotFound => ErrorCode::CudfInvalidInput,
        cudf::ErrorKind::Unsupported => ErrorCode::CudfUnsupported,
        cudf::ErrorKind::DependencyFailure => ErrorCode::CudfDependencyFailure,
        cudf::ErrorKind::InvariantViolation | cudf::ErrorKind::ExecutionContext => ErrorCode::CudfInvariantViolation,
        _ => ErrorCode::CudfDependencyFailure,
    }
}

// Native-to-adapter mapping is intentionally lossy at the top-level code:
// native `UnsupportedPlanShape` becomes adapter `UnsupportedOperator` because
// the adapter has no shape-specific code. The exact native code remains
// available in the `native_code` fact attached by `Error::from_native`.
fn code_for_native_error(error: &cudf_query_engine::QueryError) -> ErrorCode {
    match error.code() {
        cudf_query_engine::QueryErrorCode::UnsupportedPlanShape => ErrorCode::UnsupportedOperator,
        cudf_query_engine::QueryErrorCode::UnsupportedOperator => ErrorCode::UnsupportedOperator,
        cudf_query_engine::QueryErrorCode::UnsupportedExpression => ErrorCode::UnsupportedExpression,
        cudf_query_engine::QueryErrorCode::UnsupportedType => ErrorCode::UnsupportedType,
        cudf_query_engine::QueryErrorCode::InvalidPlan => ErrorCode::InvalidExecutionPlan,
        cudf_query_engine::QueryErrorCode::CudfInvalidInput => ErrorCode::CudfInvalidInput,
        cudf_query_engine::QueryErrorCode::CudfUnsupported => ErrorCode::CudfUnsupported,
        cudf_query_engine::QueryErrorCode::CudfResourceExhausted => ErrorCode::CudfResourceExhausted,
        cudf_query_engine::QueryErrorCode::CudfTemporaryFailure => ErrorCode::CudfTemporaryFailure,
        cudf_query_engine::QueryErrorCode::CudfPersistentFailure => ErrorCode::CudfPersistentFailure,
        cudf_query_engine::QueryErrorCode::CudfDependencyFailure => ErrorCode::CudfDependencyFailure,
        cudf_query_engine::QueryErrorCode::CudfInvariantViolation => ErrorCode::CudfInvariantViolation,
        cudf_query_engine::QueryErrorCode::MemoryReservationFailed => ErrorCode::MemoryReservationFailed,
        cudf_query_engine::QueryErrorCode::Internal => ErrorCode::Internal,
        _ => ErrorCode::Internal,
    }
}

fn quote(value: &str) -> String {
    if value.chars().all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.' | ':' | '/' | '(' | ')')) {
        value.to_string()
    } else {
        format!("{value:?}")
    }
}
